"""Scenario-driven synthetic forensic corpus generator (requirement T0.1).

Builds a realistic, *diverse* corpus by modelling distinct CASE SCENARIOS rather
than stitching a handful of snippets:

    * ~n/4 cases, each with structured facts (crime, location, date, time, victim,
      suspect, vehicle, witnesses, officer, evidence) drawn from large entity pools.
    * Per case it emits several INTERNALLY-CONSISTENT documents (two witness
      statements, an incident report, an interview transcript) so ``case_id`` genuinely
      groups a case's documents — the law-enforcement scenario the brief describes.
    * Documents are multi-paragraph (reports run several hundred words → multi-chunk),
      with high entity/template variety → low duplication.
    * Each case carries UNIQUE "signature" details placed in specific documents:
        - a distinctive vehicle  → primary witness statement
        - a unique evidence item → incident report
        - a memorable named person → interview transcript
      These drive a diverse ``ground_truth.json`` (one query per case, rotating the
      target doc type, with a mix of semantic / doc_type / case_id / date-range filters).

Deterministic for a fixed seed. Format writers (txt/pdf/json/eml) are isolated from
content rendering, so adding a format or a doc type stays a local change.
"""

from __future__ import annotations

import datetime
import json
from dataclasses import dataclass, field
from email.message import EmailMessage
from email.utils import format_datetime
from math import ceil
from pathlib import Path
from random import Random
from typing import Any, Callable
from xml.sax.saxutils import escape

DOC_TYPES = ("witness_statement", "report", "transcript")
FORMATS = ("txt", "pdf", "json", "eml")  # required txt/pdf/json + email evidence
_DOCS_PER_CASE = 4
_MIN_DOCS = 40  # >= 10 cases so ground_truth has >= 10 pairs

_DATE_START = datetime.date(2023, 6, 1)
_DATE_END = datetime.date(2024, 12, 31)

# ── entity pools (variety = low duplication) ─────────────────────────────────
FIRST_NAMES = [
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael", "Linda", "David",
    "Elizabeth", "William", "Susan", "Richard", "Jessica", "Joseph", "Karen", "Thomas", "Sarah",
    "Charles", "Nancy", "Daniel", "Lisa", "Matthew", "Margaret", "Anthony", "Sandra", "Mark",
    "Ashley", "Steven", "Emily", "Paul", "Donna", "Andrew", "Michelle", "Joshua", "Carol",
    "Kenneth", "Amanda", "Kevin", "Dorothy", "Brian", "Melissa", "George", "Deborah", "Edward",
    "Stephanie", "Ronald", "Rebecca", "Dario", "Marcus",
]
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez",
    "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor",
    "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez",
    "Clark", "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King", "Wright",
    "Scott", "Torres", "Nguyen", "Hill", "Flores", "Green", "Adams", "Nelson", "Baker", "Hall",
    "Rivera", "Campbell", "Mitchell", "Carter", "Roberts",
]
STREETS = [
    "Maple Street", "Oak Avenue", "Harbor Road", "Birch Lane", "Cedar Court", "Elm Street",
    "Pine Boulevard", "Riverside Drive", "Sunset Terrace", "Lincoln Avenue", "Market Street",
    "Bridge Road", "Willow Way", "Chapel Lane", "Station Road", "Kingfisher Close", "Beacon Hill",
    "Greenfield Road", "Ashford Street", "Quarry Lane", "Camden Row", "Falcon Street", "Mill Road",
    "Victoria Street", "Garrison Lane", "Old Mill Road", "Templar Street", "Hawthorn Drive",
    "Selby Road", "Marsh Lane",
]
PREMISES = [
    "a convenience store", "a residential garage", "an office building", "a distribution warehouse",
    "a multi-storey car park", "a pharmacy", "an electronics shop", "a jewellery store",
    "a corner shop", "a self-storage facility", "a ground-floor flat", "a builders' merchant",
]
VEHICLE_COLORS = ["blue", "red", "silver", "black", "white", "green", "grey", "dark blue",
                  "maroon", "beige", "navy", "gunmetal"]
VEHICLE_MAKES = ["Toyota", "Ford", "Honda", "Volkswagen", "Nissan", "Hyundai", "Kia", "Mazda",
                 "Chevrolet", "Subaru", "Renault", "Peugeot", "Vauxhall", "Skoda", "Fiat"]
VEHICLE_TYPES = ["sedan", "hatchback", "SUV", "pickup truck", "panel van", "coupe", "estate",
                 "crossover"]
VEH_DISTINCTIVE = [
    "a dented rear bumper", "a cracked windscreen", "a ski rack on the roof", "out-of-state plates",
    "a custom skull decal on the bonnet", "mismatched hubcaps", "a large rust patch on the driver's door",
    "heavily tinted rear windows", "a faded blue tarpaulin in the bed", "a missing wing mirror",
    "a bumper sticker reading COASTAL", "a roof-mounted light bar", "a cracked tail light",
    "oversized off-road tyres", "a primer-grey replacement door", "a trailer hitch",
    "peeling window film", "a chrome bull bar", "a sun-bleached bonnet", "a spare wheel on the rear",
]
EV_ADJ = ["a monogrammed", "a torn", "a half-burned", "a mud-caked", "a custom-engraved",
          "a partially melted", "a hand-stitched", "a rusted", "a brand-new", "a blood-stained"]
EV_ITEM = ["brass lighter", "leather glove", "canvas tool bag", "claw hammer", "set of bolt cutters",
           "car key fob", "prepaid mobile phone", "diner receipt", "coil of green nylon rope",
           "size-11 work boot", "balaclava", "flat-head screwdriver", "crowbar", "nylon backpack",
           "pair of pliers", "wristwatch", "baseball cap", "rubber torch", "roll of duct tape",
           "leather wallet"]
CRIME_TYPES = ["a residential burglary", "a vehicle theft", "an armed robbery", "a commercial break-in",
               "a theft from a motor vehicle", "an act of vandalism", "a shoplifting incident",
               "an attempted arson", "a street mugging", "a warehouse break-in"]
TIMES = ["shortly after 9 p.m.", "around 2:15 a.m.", "just before midnight", "at approximately 7:40 a.m.",
         "in the early hours of the morning", "around 4 p.m.", "just after sunset", "near 11:30 p.m.",
         "in the mid-afternoon", "around dawn"]
ACTIVITIES = ["walking my dog", "returning home from a late shift", "waiting at the bus stop",
              "closing up my shop", "parking my car", "jogging along the street", "unloading groceries",
              "out on my balcony", "cycling home", "locking up the office"]
SUSPECT_BUILD = ["a tall, heavyset man", "a slim woman of average height", "two young men",
                 "a man of medium build", "a short, wiry individual", "a heavily-built man",
                 "a teenager in a tracksuit", "a woman with closely cropped hair"]
SUSPECT_CLOTHES = ["a grey hooded sweatshirt", "a high-visibility work jacket", "a long dark overcoat",
                   "a red windbreaker", "a black puffer jacket and beanie", "a denim jacket",
                   "a hooded raincoat", "a navy tracksuit"]


# ── case model ───────────────────────────────────────────────────────────────
@dataclass
class Case:
    case_id: str
    crime_type: str
    date: str
    time: str
    location: str
    premises: str
    victim: str
    officer: str
    witnesses: list[str]
    suspect_build: str
    suspect_clothes: str
    veh_phrase: str          # distinctive vehicle → witness statement signature
    ev_phrase: str           # unique evidence item → report signature
    quote_name: str          # named person → transcript signature
    evidence: list[str] = field(default_factory=list)


def _unique(rng: Random, factory: Callable[[], str], used: set[str]) -> str:
    for _ in range(2000):
        v = factory()
        if v not in used:
            used.add(v)
            return v
    raise RuntimeError("exhausted unique-value attempts (increase entity pools)")


def _name(rng: Random) -> str:
    return f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"


def _random_date(rng: Random) -> str:
    span = (_DATE_END - _DATE_START).days
    return (_DATE_START + datetime.timedelta(days=rng.randint(0, span))).isoformat()


def _month_window(iso_date: str) -> dict[str, str]:
    d = datetime.date.fromisoformat(iso_date)
    first = d.replace(day=1)
    nxt = datetime.date(d.year + (d.month == 12), (d.month % 12) + 1, 1)
    return {"gte": first.isoformat(), "lte": (nxt - datetime.timedelta(days=1)).isoformat()}


def _build_case(rng: Random, used: dict[str, set]) -> Case:
    veh = _unique(
        rng,
        lambda: f"a {rng.choice(VEHICLE_COLORS)} {rng.choice(VEHICLE_MAKES)} "
        f"{rng.choice(VEHICLE_TYPES)} with {rng.choice(VEH_DISTINCTIVE)}",
        used["veh"],
    )
    ev = _unique(rng, lambda: f"{rng.choice(EV_ADJ)} {rng.choice(EV_ITEM)}", used["ev"])
    case = Case(
        case_id=_unique(rng, lambda: f"{rng.choice([2022, 2023, 2024])}-{rng.randint(1000, 9999)}", used["case"]),
        crime_type=rng.choice(CRIME_TYPES),
        date=_random_date(rng),
        time=rng.choice(TIMES),
        location=rng.choice(STREETS),
        premises=rng.choice(PREMISES),
        victim=_name(rng),
        officer=f"Officer {rng.choice(LAST_NAMES)}",
        witnesses=[_name(rng), _name(rng)],
        suspect_build=rng.choice(SUSPECT_BUILD),
        suspect_clothes=rng.choice(SUSPECT_CLOTHES),
        veh_phrase=veh,
        ev_phrase=ev,
        quote_name=_unique(rng, lambda: _name(rng), used["name"]),
    )
    case.evidence = [case.ev_phrase,
                     f"{rng.choice(EV_ADJ)} {rng.choice(EV_ITEM)}",
                     f"{rng.choice(EV_ADJ)} {rng.choice(EV_ITEM)}"]
    return case


# ── document renderers (one per doc_type; reference the case facts) ───────────
def _render_witness(rng: Random, case: Case, *, primary: bool) -> str:
    who = case.witnesses[0] if primary else case.witnesses[1]
    paras = [
        rng.choice([
            f"My name is {who}. I am providing this statement regarding {case.crime_type} "
            f"that I witnessed on {case.date} near {case.location}.",
            f"On {case.date}, {case.time}, I was {rng.choice(ACTIVITIES)} on {case.location} "
            f"when I became aware of {case.crime_type} taking place at {case.premises}.",
        ]),
        f"I saw {case.suspect_build} wearing {case.suspect_clothes}. " + rng.choice([
            "They were acting nervously and kept glancing over their shoulder.",
            "They moved quickly and seemed anxious to leave the area.",
            "At first I assumed they belonged there, but their behaviour made me suspicious.",
            "They were carrying something bulky and trying not to be noticed.",
        ]),
    ]
    if primary:
        paras.append(
            f"A vehicle was waiting a short distance away. The vehicle involved was {case.veh_phrase}. "
            "I am confident about these details because it passed directly beneath a streetlight as it left."
        )
    else:
        paras.append(rng.choice([
            "My view of any vehicle was partially blocked, so I cannot describe it reliably.",
            "I did not see a vehicle clearly, but I heard an engine revving nearby moments later.",
        ]))
    paras.append(rng.choice([
        f"The individual then left in the direction of {rng.choice(STREETS)}. I called the police immediately.",
        f"After a few moments they fled on foot toward {rng.choice(STREETS)}. I waited for officers and reported what I had seen.",
    ]))
    body = "\n\n".join(paras)
    closing = rng.choice([
        "I confirm that the above statement is true to the best of my recollection.",
        "This statement is true and accurate as far as I can recall the events.",
        "I make this statement believing it to be true and am willing to attend court if required.",
    ])
    return (
        f"WITNESS STATEMENT\nCase {case.case_id}    Date: {case.date}\n\n"
        f"{body}\n\n{closing}\nSigned: {who}"
    )


def _render_report(rng: Random, case: Case) -> str:
    evidence_block = "\n".join(f"  - {item}" for item in case.evidence)
    w1, w2 = case.witnesses
    paras = [
        f"Summary: On {case.date} at {case.time}, officers attended {case.premises} on "
        f"{case.location} following a report of {case.crime_type}. The complainant was "
        f"identified as {case.victim}.",
        rng.choice([
            f"Background: {case.victim} stated that the premises had been secure prior to the "
            f"incident, and that there was no prior history of similar offences at this address. "
            f"The surrounding area is primarily residential with limited overnight footfall.",
            f"Background: The location is {case.premises} on {case.location}. {case.victim} "
            f"reported that nothing had appeared disturbed earlier in the day. Lighting in the "
            f"vicinity is provided by two street lamps, one of which was reported as faulty.",
        ]),
        f"Scene examination: On arrival, {case.officer} secured the cordon and conducted an "
        + rng.choice([
            "initial walk-through. Point of entry appeared to be at the rear, where the fastening "
            "showed signs of having been forced. Drawers and storage units had been opened and "
            "their contents disturbed.",
            "examination of the perimeter. A ground-floor window had been levered open and glass "
            "fragments were present on the interior floor. Several surfaces were dusted for marks.",
        ]),
        f"Witness accounts: {w1} provided a statement describing {case.suspect_build} wearing "
        f"{case.suspect_clothes}, seen at the scene {case.time}. " + rng.choice([
            f"A second witness, {w2}, corroborated the timing but could not describe the suspect in detail.",
            f"{w2} reported hearing raised voices and an engine shortly before the suspect left the area.",
        ]),
        rng.choice([
            "Modus operandi: The method used in this offence is consistent with a small number of "
            "recent incidents in the wider district, where access was gained quickly and high-value, "
            "portable items were targeted before the offender left in a waiting vehicle. Intelligence "
            "checks against those reports are being carried out as part of this investigation.",
            f"Context: This is the second report of {case.crime_type} on or near {case.location} in "
            f"recent months. Patrol patterns for the area are being reviewed, and {case.officer} has "
            f"flagged the location for additional overnight attention pending the outcome of enquiries.",
        ]),
        rng.choice([
            f"Property and loss: {case.victim} reported that a number of items were taken or damaged "
            "during the incident. A preliminary list has been recorded pending a full inventory, and "
            "the estimated value is to be confirmed.",
            f"Property and loss: {case.victim} provided an initial account of missing property. The "
            "full extent of the loss has not yet been established and a detailed inventory will follow "
            "once the scene has been released.",
        ]),
        f"Evidence collected:\n{evidence_block}",
        rng.choice([
            f"Lines of enquiry: The description of the suspect and the associated vehicle have been "
            f"circulated to patrols in the {case.location} area. {case.officer} will review any "
            "matching intelligence reports and follow up with the witnesses to clarify the timeline.",
            f"Lines of enquiry: House-to-house enquiries are being conducted along {case.location}. "
            "The recovered exhibits will be cross-referenced against similar recent offences in the "
            "district, and any forensic matches will be prioritised for follow-up.",
        ]),
        rng.choice([
            f"Actions taken: The scene was cordoned and examined by the forensic team. {case.officer} "
            f"canvassed the immediate area for CCTV and additional witnesses, and exhibits were logged "
            f"into evidence and photographed in situ.",
            f"Actions taken: Photographs were taken in situ and all exhibits were logged into evidence. "
            f"{case.officer} arranged for door-to-door enquiries on {case.location} and requested "
            f"footage from nearby premises.",
        ]),
        rng.choice([
            "Follow-up: Recovered exhibits have been submitted for analysis. Enquiries are ongoing and "
            "the case remains open pending the results.",
            "Follow-up: CCTV from neighbouring premises is being requested and witness contact details "
            "have been recorded for follow-up interviews. The case remains open.",
        ]),
        f"Reporting officer: {case.officer}.",
    ]
    return f"INCIDENT REPORT\nCase {case.case_id}    Date: {case.date}\n\n" + "\n\n".join(paras)


def _render_transcript(rng: Random, case: Case) -> str:
    subject = case.witnesses[0]
    opener = rng.choice([
        f"For the record, can you state your name and what you saw on {case.date}?",
        f"Thank you for coming in. Can you tell me, in your own words, what happened on {case.date}?",
        f"Let's start from the beginning. What did you witness on {case.date}?",
    ])
    turns = [
        f"Officer ({case.officer}): {opener}",
        f"Witness ({subject}): Yes. I saw {case.crime_type} near {case.location}, {case.time}.",
        "Officer: Can you describe the person involved?",
        f"Witness: It was {case.suspect_build} wearing {case.suspect_clothes}.",
        rng.choice([
            "Officer: Did they say anything you remember?",
            "Officer: Was anyone else mentioned during the incident?",
        ]),
        f"Witness: He kept bringing up someone called {case.quote_name}. It stuck with me.",
        "Officer: Is there anything else you would like to add?",
        rng.choice([
            "Witness: Only that it happened very fast. I called it in straight away.",
            "Witness: No, that is everything I can recall clearly.",
        ]),
    ]
    return (
        f"INTERVIEW TRANSCRIPT\nCase {case.case_id}    Date: {case.date}\n\n" + "\n".join(turns)
    )


# ── format writers (content-agnostic) ────────────────────────────────────────
def _write_txt(path: Path, text: str, meta: dict) -> None:
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, text: str, meta: dict) -> None:
    payload = {"content": text, **{k: meta[k] for k in ("doc_type", "case_id", "date", "title")}}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_pdf(path: Path, text: str, meta: dict) -> None:
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    styles = getSampleStyleSheet()
    flow: list[Any] = []
    for line in text.split("\n"):
        flow.append(Spacer(1, 8) if not line.strip() else Paragraph(escape(line), styles["Normal"]))
    SimpleDocTemplate(str(path), pagesize=LETTER).build(flow)


def _write_eml(path: Path, text: str, meta: dict) -> None:
    msg = EmailMessage()
    msg["From"] = "records.officer@precinct.example"
    msg["To"] = "case.file@precinct.example"
    msg["Subject"] = f"{meta['title']} [{meta['case_id']}]"
    msg["Date"] = format_datetime(datetime.datetime.fromisoformat(f"{meta['date']}T12:00:00"))
    msg["X-Case-ID"] = meta["case_id"]
    msg["X-Doc-Type"] = meta["doc_type"]
    msg.set_content(text)
    path.write_text(msg.as_string(), encoding="utf-8")


_WRITERS: dict[str, Callable[[Path, str, dict], None]] = {
    "txt": _write_txt, "pdf": _write_pdf, "json": _write_json, "eml": _write_eml,
}

_RENDERERS: dict[str, Callable[..., str]] = {
    "witness_statement": _render_witness, "report": _render_report, "transcript": _render_transcript,
}

# Which signature each case role carries, and the ground-truth query for it.
_ROTATION = [("veh", "witness_statement"), ("ev", "report"), ("quote", "transcript")]


def _query_for(role: str, case: Case) -> str:
    if role == "veh":
        return f"vehicle described as {case.veh_phrase}"
    if role == "ev":
        return f"{case.ev_phrase} recovered from the scene"
    return f"the interview in which the suspect mentioned {case.quote_name}"


# ── orchestrator ─────────────────────────────────────────────────────────────
def generate(n: int = 120, seed: int = 42, out_dir: str | Path = "data/generated") -> dict[str, Any]:
    """Generate ``n`` documents across ``ceil(n/4)`` cases + ``ground_truth.json``.

    Returns stats: ``{out_dir, documents, cases, by_format, by_doc_type, ground_truth_pairs}``.
    Deterministic for a fixed ``seed``.
    """
    if n < _MIN_DOCS:
        raise ValueError(f"n must be >= {_MIN_DOCS} (>= 10 cases for a meaningful ground truth)")

    rng = Random(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Clean slate so the corpus is exactly `n` docs and reproducible.
    for old in out.iterdir():
        if old.is_file() and (old.suffix.lower() in {f".{f}" for f in FORMATS} or old.name == "ground_truth.json"):
            old.unlink()

    used = {"case": set(), "veh": set(), "ev": set(), "name": set()}
    cases = [_build_case(rng, used) for _ in range(ceil(n / _DOCS_PER_CASE))]

    # Per case, plan its documents (role tags mark which doc carries which signature).
    specs: list[tuple[int, Case, str, str]] = []
    for ci, case in enumerate(cases):
        specs.append((ci, case, "witness_statement", "veh"))       # primary witness → vehicle sig
        specs.append((ci, case, "report", "ev"))                   # report → evidence sig
        specs.append((ci, case, "transcript", "quote"))            # transcript → named person sig
        specs.append((ci, case, "witness_statement", "secondary")) # second witness, no signature
    specs = specs[:n]

    by_format = {f: 0 for f in FORMATS}
    by_doc_type = {d: 0 for d in DOC_TYPES}
    role_file: dict[tuple[int, str], str] = {}

    for gi, (ci, case, doc_type, role) in enumerate(specs):
        # Offset by case index so format is NOT locked to a doc's role/position
        # (otherwise every report would be a PDF, every transcript JSON, ...).
        # Still yields an even split and one of each format per 4-doc case.
        fmt = FORMATS[(gi + ci) % len(FORMATS)]
        if doc_type == "witness_statement":
            text = _render_witness(rng, case, primary=(role == "veh"))
        else:
            text = _RENDERERS[doc_type](rng, case)
        title = f"{doc_type.replace('_', ' ').title()} - Case {case.case_id}"
        meta = {"doc_type": doc_type, "case_id": case.case_id, "date": case.date, "title": title}
        filename = f"{doc_type}__{case.case_id}__{case.date}__{gi:03d}-{role}.{fmt}"
        _WRITERS[fmt](out / filename, text, meta)
        by_format[fmt] += 1
        by_doc_type[doc_type] += 1
        role_file[(ci, role)] = filename

    # Ground truth: one query per case, rotating the target doc type; unambiguous
    # because each signature is unique to exactly one document.
    ground_truth: list[dict[str, Any]] = []
    for ci, case in enumerate(cases):
        role, doc_type = _ROTATION[ci % len(_ROTATION)]
        filename = role_file.get((ci, role))
        if filename is None:  # case truncated by `n`; skip
            continue
        bucket = ci % 4
        if bucket == 0:
            filters: dict[str, Any] = {}
        elif bucket == 1:
            filters = {"doc_type": doc_type}
        elif bucket == 2:
            filters = {"case_id": case.case_id}
        else:
            filters = {"doc_type": doc_type, "date": _month_window(case.date)}
        ground_truth.append(
            {"query": _query_for(role, case), "expected_source_file": filename, "filters": filters}
        )

    (out / "ground_truth.json").write_text(
        json.dumps(ground_truth, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return {
        "out_dir": str(out),
        "documents": len(specs),
        "cases": len(cases),
        "by_format": by_format,
        "by_doc_type": by_doc_type,
        "ground_truth_pairs": len(ground_truth),
    }
