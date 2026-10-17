"""Conservative, scalable policy/stance admission of party-labeled tweets.

This is a deterministic weak-data filter, not a validated politicalness model.
It does not verify that an author's party matches the viewpoint of a passage.
Raw examples and quarantine stay under gitignored data/.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re

VERSION = "policy-stance-v1"
DOMAINS = {
    "tax_budget": r"\b(?:tax(?:es|ation)?|taxpayer\w*|federal budget|deficit|debt ceiling|government spending|federal spending|sales tax)\b",
    "healthcare": r"\b(?:health ?care|medicare|medicaid|affordable care act|obamacare|prescription drug\w*|drug prices|health insurance|reproductive rights|abortion|roe v\.? wade|maternal mortality|vaccine mandate\w*)\b",
    "immigration": r"\b(?:immigra\w*|border security|border wall|asylum|citizenship|deport\w*|undocumented|daca)\b",
    "environment_energy": r"\b(?:climate change|climate crisis|carbon emissions|fossil fuels|renewable energy|clean energy|oil and gas|drilling|green new deal|pipeline|environmental protection)\b",
    "labor_economy": r"\b(?:minimum wage|living wage|paid leave|collective bargaining|labor union\w*|workers.? rights|unemployment benefits|economic inequality|social security|food stamps|student debt|student loans|housing affordability|affordable housing)\b",
    "justice_rights": r"\b(?:gun\w*|firearm\w*|second amendment|2nd amendment|voting rights|voter suppression|voter id|police reform|criminal justice|death penalty|mass incarceration|civil rights|lgbt\w*|transgender|marriage equality|religious freedom|religious liberty|election integrity|campaign finance|dark money)\b",
    "education": r"\b(?:public schools|public education|school choice|charter schools|education funding|school vouchers|tuition.free|free college)\b",
    "foreign_defense": r"\b(?:national security|military spending|defense spending|defense budget|nato|foreign aid|sanctions|trade agreement\w*|tariff\w*|nuclear deal|iran deal|troop withdrawal|war in ukraine|war in afghanistan)\b",
    "government": r"\b(?:filibuster|supreme court reform|term limits|government shutdown|campaign contributions|corporate donations|infrastructure bill|infrastructure investment|covid.{0,4}19 relief|relief package|pathway to citizenship)\b",
}
DOMAIN_PATTERNS = {k: re.compile(v, re.I) for k, v in DOMAINS.items()}
STANCE = re.compile(
    r"\b(?:must|should|need(?:s)? to|cannot|can't|unacceptable|unconstitutional|"
    r"support(?:s|ed|ing)?|oppos(?:e|es|ed|ing)|reject(?:s|ed|ing)?|"
    r"fight(?:ing)? for|fight(?:ing)? against|protect(?:s|ed|ing)?|"
    r"defend(?:s|ed|ing)?|expand(?:s|ed|ing)?|invest(?:s|ed|ing)?|"
    r"repeal(?:s|ed|ing)?|abolish\w*|ban(?:s|ned|ning)?|"
    r"cut(?:s|ting)?|fund(?:s|ed|ing)?|increase(?:s|d)?|reduce(?:s|d)?|"
    r"lower(?:s|ed|ing)?|raise(?:s|d)?|restore(?:s|d)?|"
    r"voted (?:for|against)|vote (?:for|against)|"
    r"co.?sponsor\w*|introduc(?:e|ed|ing)|"
    r"deserve(?:s|d)?|fair share|devastat\w*|harmful|reckless|"
    r"disastrous|shameful|critical|crucial|essential)\b", re.I)
CEREMONIAL = re.compile(
    r"^\W*(?:happy\b|congratulations\b|congrats\b|thank you\b|"
    r"thanks to\b|my (?:heart|prayers|condolences)\b|"
    r"our (?:hearts|prayers|condolences)\b|on this day\b|"
    r"wishing\b|rest in peace\b)", re.I)
SERVICE_TEMPLATE = re.compile(
    r"\b(?:sign up for my|subscribe to my|call my office|contact my office|"
    r"my office can help|join (?:me|us) (?:at|for)|tune in (?:at|to)|"
    r"watch (?:me|my interview)|donate (?:now|today|to my)|"
    r"register (?:for|at)|rsvp|office hours|telephone town hall)\b", re.I)


def classify(row):
    """Return exclusion reason (or None) and matched policy domains."""
    text = row["text"].strip()
    if row.get("label") not in (0, 1) or row.get("label_kind") != "weak":
        return "invalid_label_metadata", []
    if not row.get("group_ids") or not all(g.startswith("twitter:author:") for g in row["group_ids"]):
        return "missing_author_group", []
    if not 20 <= len(text.split()) <= 150:
        return "length", []
    if re.match(r"^\s*RT\b", text):
        return "retweet", []
    # Truncation often removes the claim, its negation or its target.
    if re.search(r"(?:…|\.{3})\s*$", text):
        return "truncated", []
    if CEREMONIAL.search(text):
        return "ceremonial_prefix", []
    if SERVICE_TEMPLATE.search(text):
        return "service_or_promotion_template", []
    domains = [key for key, pattern in DOMAIN_PATTERNS.items() if pattern.search(text)]
    if not domains:
        return "no_explicit_policy_domain", []
    if not STANCE.search(text):
        return "no_explicit_stance_cue", domains
    return None, domains


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/available/Jacobvs_PoliticalTweets/candidates.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/tweet-admission-v1"))
    parser.add_argument("--report", type=Path, default=Path("reports/tweet-admission.json"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    counts, labels, domains = Counter(), Counter(), Counter()
    eligible_groups, all_groups = set(), set()
    # A deterministic reservoir supports local review without a public text dump.
    sample, rng = [], random.Random(42)
    with args.input.open() as src, (args.output / "eligible.jsonl").open("w") as admitted, (args.output / "quarantine.jsonl").open("w") as quarantine:
        for line in src:
            row = json.loads(line)
            counts["input"] += 1
            all_groups.update(row.get("group_ids", []))
            reason, matched = classify(row)
            row.update(admission_rule_version=VERSION, politicalness_verified=False,
                       label_verified=False, policy_domains=matched)
            if reason:
                counts[reason] += 1
                row.update(admission="quarantined", admission_exclusion=reason)
                handle = quarantine
            else:
                counts["eligible"] += 1
                labels[str(row["label"])] += 1
                domains.update(matched)
                eligible_groups.update(row["group_ids"])
                row["admission"] = "heuristic_policy_and_stance"
                handle = admitted
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            if len(sample) < 300:
                sample.append(row)
            else:
                index = rng.randrange(counts["input"])
                if index < len(sample):
                    sample[index] = row
    (args.output / "local-audit-sample.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sample))
    report = {
        "source": "Jacobvs/PoliticalTweets", "rule_version": VERSION,
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "counts": dict(counts), "eligible_labels": dict(labels),
        "eligible_author_groups": len(eligible_groups), "input_author_groups": len(all_groups),
        "eligible_policy_domain_counts_nonexclusive": dict(domains),
        "paths": {"eligible": str(args.output / "eligible.jsonl"), "quarantine": str(args.output / "quarantine.jsonl")},
        "output_sha256": {name: hashlib.sha256((args.output / name).read_bytes()).hexdigest() for name in ("eligible.jsonl", "quarantine.jsonl")},
        "policy": "Require an explicit policy-domain match plus a stance cue; reject truncated endings, retweets, ceremonial prefixes, service/promotion templates, invalid metadata and out-of-range lengths. Preserve all author group IDs. No label is changed.",
        "limitations": [
            "Party affiliation remains a weak proxy for expressed ideology, including for eligible passages.",
            "Politicalness and viewpoint are not human verified; admission precision and recall have not been measured.",
            "A stance cue can occur in a quotation or unrelated clause; sarcasm, negation and topic-label confounding remain.",
            "The English policy lexicon is selective and favors explicit vocabulary. Quarantine is not a nonpolitical ground-truth class.",
            "All cross-source deduplication and author-separated split assignment must happen after this admission step.",
            "Eligible data belongs only in an identifiable weak-supervision tier, not a human-verified evaluation set.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
