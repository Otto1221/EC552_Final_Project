"""
Scrape real genetic circuits from SynBioHub.
Converts SBOL XML data into training JSON format for fine-tuning.

Usage:
    python scrape_circuits.py --output-dir ./finetune/scraped
"""

import json
import time
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote

import requests

# ---------------------------------------------------------------------------
# SynBioHub config
# ---------------------------------------------------------------------------

SYNBIOHUB = "https://synbiohub.org"

# Collections known to contain full circuit designs with subcomponents
COLLECTION_URIS = [
    # Cello designs — full circuits with ModuleDefinitions and interactions
    f"{SYNBIOHUB}/public/Eco2C1G5T1/Eco2C1G5T1_collection/1",
    # DataCurationProject — 70 curated plasmids
    f"{SYNBIOHUB}/public/DataCurationProject/DataCurationProject_collection/1",
    # iGEM 2016 interlab devices
    f"{SYNBIOHUB}/public/iGEM_2016_interlab/iGEM_2016_interlab_collection/1",
    # Digitalizer circuits
    f"{SYNBIOHUB}/public/Digitalizer/Digitalizer_collection/1",
]

# Search queries to find individual circuits (with subcomponents)
SEARCH_QUERIES = [
    "toggle switch", "repressilator", "oscillator", "biosensor",
    "logic gate", "AND gate", "NOR gate", "NOT gate", "inverter",
    "kill switch", "CRISPR", "CRISPRi", "quorum sensing",
    "GFP expression", "T7 expression", "lac operon", "tet repressor",
    "genetic circuit", "feedback loop", "cascade", "amplifier",
    "band pass", "pulse generator", "memory circuit", "recombinase",
    "riboswitch", "toehold switch", "metabolic pathway",
    "fluorescent reporter", "mCherry", "toxin antitoxin",
]

# SBOL2 namespaces
SBOL = 'http://sbols.org/v2#'
RDF = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#'
DC = 'http://purl.org/dc/terms/'

# SO roles mapping
SO_ROLES = {
    'http://identifiers.org/so/SO:0000167': 'promoter',
    'http://identifiers.org/so/SO:0000139': 'rbs',
    'http://identifiers.org/so/SO:0000316': 'cds',
    'http://identifiers.org/so/SO:0000141': 'terminator',
    'http://identifiers.org/so/SO:0000057': 'operator',
    'http://identifiers.org/so/SO:0000552': 'rbs',
    'http://identifiers.org/so/SO:0000110': 'other',
    'http://identifiers.org/so/SO:0000804': None,  # engineered_region = top-level circuit
    'http://identifiers.org/so/SO:0000296': 'other',  # origin_of_replication
    'http://identifiers.org/so/SO:0005850': 'other',  # primer_binding_site
    'http://identifiers.org/so/SO:0001977': 'rbs',  # ribozyme insulator (RiboJ = synthetic RBS)
}

# SBO interaction types
SBO_INTERACTIONS = {
    'http://identifiers.org/biomodels.sbo/SBO:0000169': 'repression',
    'http://identifiers.org/biomodels.sbo/SBO:0000170': 'activation',
    'http://identifiers.org/biomodels.sbo/SBO:0000589': 'transcription',
    'http://identifiers.org/biomodels.sbo/SBO:0000584': 'translation',
    'http://identifiers.org/biomodels.sbo/SBO:0000168': 'repression',
    'http://identifiers.org/biomodels.sbo/SBO:0000459': 'activation',
    'http://identifiers.org/biomodels.sbo/SBO:0000642': 'repression',
    'http://identifiers.org/biomodels.sbo/SBO:0000020': 'repression',
    'http://identifiers.org/biomodels.sbo/SBO:0000179': None,  # degradation — skip
}

# Name-based type inference for parts missing SO roles
NAME_TYPE_HINTS = {
    'promoter': 'promoter', 'prom': 'promoter', 'plac': 'promoter', 'ptet': 'promoter',
    'pbad': 'promoter', 'pt7': 'promoter', 'plux': 'promoter', 'ptrc': 'promoter',
    'rbs': 'rbs', 'riboj': 'rbs', 'shine': 'rbs', 'b003': 'rbs',
    'terminator': 'terminator', 'term': 'terminator', 'b0015': 'terminator',
    'b0010': 'terminator', 'b0012': 'terminator',
    'operator': 'operator', 'laco': 'operator', 'teto': 'operator', 'arabad': 'operator',
    'gfp': 'cds', 'yfp': 'cds', 'rfp': 'cds', 'cfp': 'cds', 'mcherry': 'cds',
    'laci': 'cds', 'tetr': 'cds', 'luxr': 'cds', 'luxi': 'cds', 'arac': 'cds',
    'ampr': 'cds', 'kanr': 'cds', 'cmr': 'cds',
    'orit': 'other', 'ori': 'other', 'r6k': 'other', 'attp': 'other',
}


def infer_type_from_name(name: str) -> str:
    """Infer component type from its name when SO role is missing."""
    name_lower = name.lower().replace(' ', '').replace('-', '').replace('_', '')
    for hint, ctype in NAME_TYPE_HINTS.items():
        if hint in name_lower:
            return ctype
    return 'other'


# ---------------------------------------------------------------------------
# SBOL XML parsing
# ---------------------------------------------------------------------------

def parse_collection_sbol(xml_text: str) -> list[dict]:
    """Parse a collection's SBOL XML, extracting all circuits with subcomponents."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        print(f"    XML parse error: {e}")
        return []

    # Build URI → ComponentDefinition map
    uri_map = {}
    for cd in root.findall(f'.//{{{SBOL}}}ComponentDefinition'):
        uri = cd.get(f'{{{RDF}}}about', '')
        uri_map[uri] = cd

    # Build URI → ModuleDefinition map
    mod_map = {}
    for md in root.findall(f'.//{{{SBOL}}}ModuleDefinition'):
        uri = md.get(f'{{{RDF}}}about', '')
        mod_map[uri] = md

    circuits = []

    # Find all ComponentDefinitions that HAVE subcomponents (= circuits)
    for cd in root.findall(f'.//{{{SBOL}}}ComponentDefinition'):
        sub_comps = cd.findall(f'{{{SBOL}}}component')
        if len(sub_comps) < 2:
            continue

        uri = cd.get(f'{{{RDF}}}about', '')
        name_el = cd.find(f'{{{DC}}}title')
        desc_el = cd.find(f'{{{DC}}}description')
        display_id = cd.find(f'{{{SBOL}}}displayId')

        circuit_name = name_el.text if name_el is not None else (
            display_id.text if display_id is not None else 'circuit')
        circuit_desc = desc_el.text if desc_el is not None else ''

        components = []
        comp_local_to_def_uri = {}  # local component URI → definition URI

        for sc in sub_comps:
            sc_inner = sc.find(f'{{{SBOL}}}Component')
            if sc_inner is None:
                continue

            def_ref = sc_inner.find(f'{{{SBOL}}}definition')
            if def_ref is None:
                continue

            local_uri = sc_inner.get(f'{{{RDF}}}about', '')
            def_uri = def_ref.get(f'{{{RDF}}}resource', '')
            comp_local_to_def_uri[local_uri] = def_uri

            ref_cd = uri_map.get(def_uri)
            if ref_cd is None:
                continue

            # Get name
            ref_name_el = ref_cd.find(f'{{{DC}}}title')
            ref_display_id = ref_cd.find(f'{{{SBOL}}}displayId')
            comp_name = ref_name_el.text if ref_name_el is not None else (
                ref_display_id.text if ref_display_id is not None else def_uri.split('/')[-1])

            # Get type from SO role
            comp_type = None
            for role in ref_cd.findall(f'{{{SBOL}}}role'):
                role_uri = role.get(f'{{{RDF}}}resource', '')
                if role_uri in SO_ROLES:
                    comp_type = SO_ROLES[role_uri]
                    break

            # If no role or role mapped to None, infer from name
            if comp_type is None:
                comp_type = infer_type_from_name(comp_name)

            # Get sequence
            sequence = None
            for seq_ref in ref_cd.findall(f'{{{SBOL}}}sequence'):
                seq_uri = seq_ref.get(f'{{{RDF}}}resource', '')
                for seq_elem in root.findall(f'.//{{{SBOL}}}Sequence'):
                    if seq_elem.get(f'{{{RDF}}}about', '') == seq_uri:
                        elements = seq_elem.find(f'{{{SBOL}}}elements')
                        if elements is not None and elements.text:
                            sequence = elements.text

            comp_id = comp_name.replace(' ', '_').replace('-', '_').lower()
            # Deduplicate IDs
            base_id = comp_id
            counter = 2
            existing_ids = {c['id'] for c in components}
            while comp_id in existing_ids:
                comp_id = f"{base_id}_{counter}"
                counter += 1

            components.append({
                'id': comp_id,
                'type': comp_type,
                'name': comp_name,
                'sequence': sequence,
                '_def_uri': def_uri,
            })

        if len(components) < 2:
            continue

        # Parse interactions from ModuleDefinitions
        interactions = parse_module_interactions(root, components, comp_local_to_def_uri)

        # If no module interactions, infer from component ordering
        if not interactions:
            interactions = infer_interactions(components)

        # Strip internal fields
        for c in components:
            c.pop('_def_uri', None)

        # Skip circuits that are just "other" parts (ori, attp, etc.)
        typed = [c for c in components if c['type'] in ('promoter', 'rbs', 'cds', 'terminator', 'operator')]
        if len(typed) < 2:
            continue

        circuits.append({
            'name': circuit_name.replace(' ', '_').replace('-', '_'),
            'description': circuit_desc,
            'components': components,
            'interactions': interactions,
            'source_uri': uri,
        })

    return circuits


def parse_module_interactions(root, components, comp_local_to_def_uri) -> list[dict]:
    """Extract interactions from ModuleDefinitions and map to component IDs."""
    interactions = []

    # Build def_uri → component id map
    def_uri_to_id = {}
    for c in components:
        def_uri_to_id[c['_def_uri']] = c['id']

    for md in root.findall(f'.//{{{SBOL}}}ModuleDefinition'):
        for inter_wrapper in md.findall(f'{{{SBOL}}}interaction'):
            inter_elem = inter_wrapper.find(f'{{{SBOL}}}Interaction')
            if inter_elem is None:
                inter_elem = inter_wrapper

            # Get interaction type
            inter_type = None
            for itype in inter_elem.findall(f'{{{SBOL}}}type'):
                type_uri = itype.get(f'{{{RDF}}}resource', '')
                if type_uri in SBO_INTERACTIONS:
                    inter_type = SBO_INTERACTIONS[type_uri]
                    break

            if inter_type is None:
                continue

            # Get participants
            from_id = None
            to_id = None
            for part in inter_elem.findall(f'.//{{{SBOL}}}Participation'):
                role_el = part.find(f'{{{SBOL}}}role')
                participant_el = part.find(f'{{{SBOL}}}participant')
                if role_el is None or participant_el is None:
                    continue

                role_uri = role_el.get(f'{{{RDF}}}resource', '')
                part_uri = participant_el.get(f'{{{RDF}}}resource', '')

                # Resolve to component ID
                def_uri = comp_local_to_def_uri.get(part_uri, part_uri)
                comp_id = def_uri_to_id.get(def_uri)

                # SBO participation roles
                role_name = role_uri.split('/')[-1]
                if any(x in role_name for x in ['0000645', '0000644', 'template']):
                    # template / reactant
                    from_id = comp_id
                elif any(x in role_name for x in ['0000011', '0000643', 'product']):
                    # product
                    to_id = comp_id
                elif 'inhibitor' in role_name or 'stimulator' in role_name or 'modifier' in role_name:
                    from_id = comp_id
                elif 'inhibited' in role_name or 'stimulated' in role_name or 'modified' in role_name:
                    to_id = comp_id

            if from_id and to_id:
                interactions.append({
                    'type': inter_type,
                    'from': from_id,
                    'to': to_id,
                })

    return interactions


def infer_interactions(components: list[dict]) -> list[dict]:
    """Infer interactions from components, regardless of order.

    Strategy:
    - Sort into biological order (promoter, operator, rbs, cds, terminator, other)
    - Then walk the sorted list to find promoter→cds and rbs→cds pairs
    - If components ARE in biological order already, the sort is a no-op
    """
    # Try to sort into biological order
    type_order = {'promoter': 0, 'operator': 1, 'rbs': 2, 'cds': 3, 'terminator': 4, 'other': 5}

    # Group by type
    promoters = [c for c in components if c['type'] == 'promoter']
    rbs_list = [c for c in components if c['type'] == 'rbs']
    cds_list = [c for c in components if c['type'] == 'cds']

    interactions = []

    # Pair promoters with CDS (transcription)
    # If multiple promoters and CDS, try to pair them
    if promoters and cds_list:
        if len(promoters) == 1:
            # Single promoter drives all CDS
            for cds in cds_list:
                interactions.append({
                    'type': 'transcription',
                    'from': promoters[0]['id'],
                    'to': cds['id'],
                })
        else:
            # Multiple promoters — pair round-robin or 1:1
            for i, cds in enumerate(cds_list):
                prom = promoters[min(i, len(promoters) - 1)]
                interactions.append({
                    'type': 'transcription',
                    'from': prom['id'],
                    'to': cds['id'],
                })

    # Pair RBS with CDS (translation)
    if rbs_list and cds_list:
        if len(rbs_list) == 1 and len(cds_list) == 1:
            interactions.append({
                'type': 'translation',
                'from': rbs_list[0]['id'],
                'to': cds_list[0]['id'],
            })
        else:
            # Pair 1:1 by position
            for i, rbs in enumerate(rbs_list):
                if i < len(cds_list):
                    interactions.append({
                        'type': 'translation',
                        'from': rbs['id'],
                        'to': cds_list[i]['id'],
                    })

    return interactions


# ---------------------------------------------------------------------------
# Scraping strategies
# ---------------------------------------------------------------------------

def scrape_collections() -> list[dict]:
    """Scrape full collections from SynBioHub."""
    all_circuits = []

    for uri in COLLECTION_URIS:
        name = uri.split('/')[-2]
        print(f"\n  Fetching collection: {name}")
        try:
            resp = requests.get(f'{uri}/sbol', timeout=120)
            if resp.status_code != 200:
                print(f"    HTTP {resp.status_code}")
                continue
            circuits = parse_collection_sbol(resp.text)
            print(f"    Found {len(circuits)} circuits")
            all_circuits.extend(circuits)
        except ET.ParseError as e:
            print(f"    XML parse error: {e}")
        except Exception as e:
            print(f"    Error: {e}")
        time.sleep(1)

    return all_circuits


def scrape_search_results() -> list[dict]:
    """Search SynBioHub and scrape individual circuits that have subcomponents."""
    all_circuits = []
    seen_uris = set()

    for query in SEARCH_QUERIES:
        print(f"  Searching: {query}")
        try:
            url = f"{SYNBIOHUB}/search/?q={quote(query)}&offset=0&limit=20"
            resp = requests.get(url, headers={'Accept': 'application/json'}, timeout=30)
            if resp.status_code != 200:
                continue
            hits = resp.json()
        except Exception as e:
            print(f"    Search error: {e}")
            continue

        time.sleep(0.5)

        for hit in hits:
            uri = hit.get('uri', '')
            if uri in seen_uris:
                continue
            seen_uris.add(uri)

            # Only ComponentDefinitions
            if 'ComponentDefinition' not in hit.get('type', ''):
                continue

            try:
                sbol_resp = requests.get(f'{uri}/sbol', timeout=10, stream=True)
                if sbol_resp.status_code != 200:
                    sbol_resp.close()
                    continue
                # Read up to 500KB, skip if larger
                chunks = []
                total = 0
                for chunk in sbol_resp.iter_content(chunk_size=65536):
                    chunks.append(chunk)
                    total += len(chunk)
                    if total > 500_000:
                        break
                sbol_resp.close()
                if total > 500_000:
                    continue
                sbol_text = b''.join(chunks).decode('utf-8', errors='replace')

                circuits = parse_collection_sbol(sbol_text)
                for c in circuits:
                    # Use hit description if circuit has none
                    if not c['description'] and hit.get('description'):
                        c['description'] = hit['description']
                    all_circuits.append(c)
                    print(f"    + {c['name']}: {len(c['components'])} parts, {len(c['interactions'])} interactions")
            except (requests.Timeout, requests.ConnectionError):
                pass
            except Exception:
                pass

            time.sleep(0.2)

            if len(all_circuits) >= 500:
                break
        if len(all_circuits) >= 500:
            break

    return all_circuits


# ---------------------------------------------------------------------------
# Description generator
# ---------------------------------------------------------------------------

def generate_description(circuit: dict) -> str:
    """Generate a natural language description from a circuit JSON."""
    desc = circuit.get('description', '')
    if desc and len(desc) > 30:
        return desc

    comp_types = {}
    for c in circuit['components']:
        comp_types.setdefault(c['type'], []).append(c['name'])

    parts = []
    if 'promoter' in comp_types:
        proms = comp_types['promoter']
        if len(proms) == 1:
            parts.append(f"driven by {proms[0]}")
        else:
            parts.append(f"with promoters: {', '.join(proms)}")

    if 'cds' in comp_types:
        genes = comp_types['cds']
        if len(genes) <= 3:
            parts.append(f"expressing {', '.join(genes)}")
        else:
            parts.append(f"expressing {len(genes)} genes including {', '.join(genes[:3])}")

    if 'operator' in comp_types:
        parts.append(f"with operator sites: {', '.join(comp_types['operator'])}")

    inter_types = set(i['type'] for i in circuit['interactions'])
    if 'repression' in inter_types and 'activation' in inter_types:
        parts.append("featuring both activation and repression interactions")
    elif 'repression' in inter_types:
        parts.append("with repression-based regulation")
    elif 'activation' in inter_types:
        parts.append("with activation-based regulation")

    return f"A genetic circuit {', '.join(parts)}." if parts else "A genetic circuit"


# ---------------------------------------------------------------------------
# Convert to training format
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert synthetic biologist. Given a natural language description of a genetic circuit, output a JSON object describing the circuit components and interactions.

The JSON must follow this schema:
{
  "name": "string — circuit name",
  "description": "string — brief description",
  "components": [
    {
      "id": "string — unique identifier (snake_case, descriptive)",
      "type": "promoter | rbs | cds | terminator | operator | other",
      "name": "string — component name",
      "sequence": "string or null — DNA sequence if known"
    }
  ],
  "interactions": [
    {
      "type": "activation | repression | transcription | translation",
      "from": "component id",
      "to": "component id"
    }
  ]
}

Component types:
- promoter: Drives transcription of downstream genes
- rbs: Ribosome binding site — enables translation of the next CDS
- cds: Coding sequence — gene that produces a protein
- terminator: Stops transcription
- operator: DNA binding site for regulatory proteins
- other: Degradation tags, linkers, origins of replication, etc.

Interaction types:
- transcription: promoter → cds (which gene a promoter drives)
- translation: rbs → cds (every RBS must translate its downstream CDS)
- activation: cds → promoter/operator (protein activates a promoter)
- repression: cds → promoter/operator (protein represses a promoter)

Respond with valid JSON only, no explanation."""


def to_training_format(circuits: list[dict]) -> list[dict]:
    """Convert scraped circuits to JSONL training format."""
    training = []
    seen_names = set()

    for circuit in circuits:
        # Skip duplicates
        name = circuit.get('name', '')
        if name in seen_names:
            continue
        seen_names.add(name)

        # Skip circuits with too few or too many components
        if len(circuit['components']) < 2 or len(circuit['components']) > 30:
            continue

        # Ensure all interactions reference valid components
        valid_ids = {c['id'] for c in circuit['components']}
        circuit['interactions'] = [
            i for i in circuit['interactions']
            if i['from'] in valid_ids and i['to'] in valid_ids
        ]

        # Generate description
        desc = generate_description(circuit)

        # Clean circuit for output (remove internal fields)
        clean = {
            'name': circuit['name'],
            'description': desc,
            'components': [{k: v for k, v in c.items() if k != 'source_uri'}
                           for c in circuit['components']],
            'interactions': circuit['interactions'],
        }

        entry = {
            'messages': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': desc},
                {'role': 'assistant', 'content': json.dumps(clean, indent=2)},
            ]
        }
        training.append(entry)

    return training


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Scrape genetic circuits from SynBioHub")
    parser.add_argument('--output-dir', type=str, default=str(Path(__file__).parent / 'scraped'))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("=== Scraping SynBioHub collections ===")
    collection_circuits = scrape_collections()

    print("\n=== Searching SynBioHub for individual circuits ===")
    search_circuits = scrape_search_results()

    # Deduplicate by name
    all_circuits = collection_circuits
    seen = {c['name'] for c in all_circuits}
    for c in search_circuits:
        if c['name'] not in seen:
            seen.add(c['name'])
            all_circuits.append(c)

    # Save raw data
    raw_path = output_dir / 'synbiohub_raw.json'
    with open(raw_path, 'w') as f:
        json.dump(all_circuits, f, indent=2, default=str)
    print(f"\nSaved {len(all_circuits)} raw circuits to {raw_path}")

    # Convert to training format
    training = to_training_format(all_circuits)

    # Save as JSONL
    training_path = output_dir / 'scraped_training.jsonl'
    with open(training_path, 'w') as f:
        for entry in training:
            f.write(json.dumps(entry) + '\n')

    print(f"\n=== SUMMARY ===")
    print(f"Total circuits found: {len(all_circuits)}")
    print(f"Valid training examples: {len(training)}")
    print(f"Saved to: {training_path}")
    print(f"\nTo merge with existing training data, run:")
    print(f"  cat finetune/scraped/scraped_training.jsonl >> finetune/train.jsonl")


if __name__ == '__main__':
    main()
