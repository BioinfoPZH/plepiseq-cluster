import argparse
import sys

import requests

SPEC = "pubmlst_campylobacter_seqdef"
DATABASE = "C. jejuni / C. coli cgMLST v2"
MAX_COLUMNS = 1143


def main():
    parser = argparse.ArgumentParser(
        description="Download Campylobacter cgMLST profiles from PubMLST."
    )
    parser.add_argument(
        "-o",
        "--output",
        default="profiles.list",
        help="Output file path (default: profiles.list)",
    )
    args = parser.parse_args()

    scheme_link = ""
    scheme_table = requests.get(f"https://rest.pubmlst.org/db/{SPEC}/schemes")
    scheme_table.raise_for_status()
    for scheme in scheme_table.json()["schemes"]:
        if DATABASE == scheme["description"]:
            scheme_link = scheme["scheme"]
            break

    if not scheme_link:
        print(f"Error: scheme '{DATABASE}' not found in {SPEC}", file=sys.stderr)
        sys.exit(1)

    print(f"Downloading Campylobacter profiles to {args.output}")
    profile = requests.get(scheme_link + "/profiles_csv")
    profile.raise_for_status()

    with open(args.output, "w") as f:
        for line in profile.iter_lines():
            fields = [x.decode("utf-8", errors="replace") for x in line.split()]
            fields = ["0" if x == "N" else x for x in fields]
            f.write("\t".join(fields[:MAX_COLUMNS]) + "\n")

    print("Done.")


if __name__ == "__main__":
    main()
