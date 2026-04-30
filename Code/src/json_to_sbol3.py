#!/usr/bin/env python3
"""
JSON → SBOL3 XML converter for the Newgenes pipeline.

Completes the end-to-end pipeline analogous to Chen & Truong (2026):
  NL description → circuit JSON → SBOL3 XML (runnable output)

The paper went from natural language → executable CC3D code.
We go from natural language → circuit JSON → SBOL3 RDF/XML.

SBOL3 spec: https://sbolstandard.org/docs/SBOL3.0specification.pdf

Usage:
  python json_to_sbol3.py circuit.json -o circuit.xml
  python json_to_sbol3.py --jsonl train.jsonl --index 37 -o toggle_switch.xml
  echo '{"components":[...],"interactions":[...]}' | python json_to_sbol3.py -o out.xml
"""
import json
import sys
import argparse
from xml.etree import ElementTree as ET
from xml.dom import minidom
import hashlib
import re

# SBOL3 namespaces
SBOL3 = "http://sbols.org/v3#"
RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
DCTERMS = "http://purl.org/dc/terms/"
PROV = "http://www.w3.org/ns/prov#"
OM = "http://www.ontology-of-units-of-measure.org/resource/om-2/"
SO = "http://identifiers.org/SO:"
SBO = "http://identifiers.org/biomodels.sbo/SBO:"

# Sequence Ontology roles for component types
SO_ROLES = {
    'promoter':   'SO:0000167',
    'rbs':        'SO:0000139',
    'cds':        'SO:0000316',
    'terminator': 'SO:0000141',
    'operator':   'SO:0000057',
    'other':      'SO:0000110',  # sequence_feature (generic)
}

# SBO interaction types
SBO_INTERACTION_TYPES = {
    'activation':        'SBO:0000170',  # stimulation
    'repression':        'SBO:0000169',  # inhibition
    'inhibition':        'SBO:0000169',  # inhibition (synonym of repression per system prompt)
    'transcription':     'SBO:0000589',  # genetic production (transcription)
    'translation':       'SBO:0000184',  # genetic production (translation)
    'production':        'SBO:0000393',  # production of a small molecule
    'complex_formation': 'SBO:0000177',  # non-covalent binding
    'degradation':       'SBO:0000179',  # degradation
}

# Participation roles
SBO_ROLES = {
    'activation': {
        'from': 'SBO:0000459',  # stimulator
        'to':   'SBO:0000643',  # stimulated
    },
    'repression': {
        'from': 'SBO:0000020',  # inhibitor
        'to':   'SBO:0000642',  # inhibited
    },
    'inhibition': {
        'from': 'SBO:0000020',  # inhibitor
        'to':   'SBO:0000642',  # inhibited
    },
    'transcription': {
        'from': 'SBO:0000645',  # template
        'to':   'SBO:0000011',  # product
    },
    'translation': {
        'from': 'SBO:0000645',  # template
        'to':   'SBO:0000011',  # product
    },
    'production': {
        'from': 'SBO:0000010',  # reactant (enzyme)
        'to':   'SBO:0000011',  # product
    },
    'complex_formation': {
        'from': 'SBO:0000280',  # ligand
        'to':   'SBO:0000280',  # ligand (both sides of a non-covalent bond)
    },
    'degradation': {
        'from': 'SBO:0000010',  # reactant (substrate being degraded)
        'to':   'SBO:0000011',  # product (degradation product)
    },
}

BASE_URI = "https://newgenes.org/"


def sanitize_uri(name):
    """Convert a component name to a valid URI fragment."""
    return re.sub(r'[^a-zA-Z0-9_]', '_', name)


def circuit_to_sbol3(circuit, circuit_name=None, description=None):
    """
    Convert a Newgenes circuit JSON to SBOL3 RDF/XML.

    Args:
        circuit: dict with 'components' and 'interactions'
        circuit_name: optional name for the top-level component
        description: optional description

    Returns:
        str: SBOL3 RDF/XML string
    """
    if not circuit_name:
        # Generate from component names
        cds_names = [c['name'] for c in circuit['components'] if c['type'] == 'cds']
        circuit_name = '_'.join(cds_names[:3]) + '_circuit' if cds_names else 'circuit'
    circuit_name = sanitize_uri(circuit_name)

    # Register namespaces
    ET.register_namespace('sbol3', SBOL3)
    ET.register_namespace('rdf', RDF)
    ET.register_namespace('dcterms', DCTERMS)
    ET.register_namespace('prov', PROV)
    ET.register_namespace('om', OM)

    # Root RDF element
    root = ET.Element(f'{{{RDF}}}RDF')

    # Top-level Component (the circuit itself)
    circuit_uri = f"{BASE_URI}{circuit_name}"
    top_comp = ET.SubElement(root, f'{{{SBOL3}}}Component')
    top_comp.set(f'{{{RDF}}}about', circuit_uri)

    # Display ID
    display_id = ET.SubElement(top_comp, f'{{{SBOL3}}}displayId')
    display_id.text = circuit_name

    # Name
    name_el = ET.SubElement(top_comp, f'{{{DCTERMS}}}title')
    name_el.text = circuit_name.replace('_', ' ').title()

    # Namespace (required by SBOL3 spec)
    ns_el = ET.SubElement(top_comp, f'{{{SBOL3}}}hasNamespace')
    ns_el.set(f'{{{RDF}}}resource', BASE_URI.rstrip('/'))

    # Description
    if description:
        desc_el = ET.SubElement(top_comp, f'{{{DCTERMS}}}description')
        desc_el.text = description

    # Type: DNA
    type_el = ET.SubElement(top_comp, f'{{{SBOL3}}}type')
    type_el.set(f'{{{RDF}}}resource', 'http://www.biopax.org/release/biopax-level3.owl#DnaRegion')  # SBOL3 standard DNA component type

    # Role: engineered_region
    role_el = ET.SubElement(top_comp, f'{{{SBOL3}}}role')
    role_el.set(f'{{{RDF}}}resource', 'SO:0000804')  # engineered_region

    # --- SubComponents ---
    comp_uri_map = {}  # name → URI for reference
    for comp in circuit['components']:
        comp_name = sanitize_uri(comp['name'])
        comp_uri = f"{circuit_uri}/{comp_name}"
        comp_uri_map[comp['name']] = comp_uri

        sub = ET.SubElement(top_comp, f'{{{SBOL3}}}hasFeature')

        sub_comp = ET.SubElement(sub, f'{{{SBOL3}}}SubComponent')
        sub_comp.set(f'{{{RDF}}}about', comp_uri)

        # Display ID
        sc_display = ET.SubElement(sub_comp, f'{{{SBOL3}}}displayId')
        sc_display.text = comp_name

        # Name
        sc_name = ET.SubElement(sub_comp, f'{{{DCTERMS}}}title')
        sc_name.text = comp['name']

        # Role (SO term)
        so_role = SO_ROLES.get(comp['type'], SO_ROLES['other'])
        sc_role = ET.SubElement(sub_comp, f'{{{SBOL3}}}role')
        sc_role.set(f'{{{RDF}}}resource', so_role)

        # Instance of (reference to a Component definition)
        # Create a top-level Component for each part type
        part_def_uri = f"{BASE_URI}parts/{comp_name}"
        instance_of = ET.SubElement(sub_comp, f'{{{SBOL3}}}instanceOf')
        instance_of.set(f'{{{RDF}}}resource', part_def_uri)

    # --- Interactions ---
    for i, ix in enumerate(circuit['interactions']):
        ix_name = f"interaction_{i}"
        ix_uri = f"{circuit_uri}/{ix_name}"

        interaction = ET.SubElement(top_comp, f'{{{SBOL3}}}hasInteraction')

        ix_el = ET.SubElement(interaction, f'{{{SBOL3}}}Interaction')
        ix_el.set(f'{{{RDF}}}about', ix_uri)

        # Display ID
        ix_display = ET.SubElement(ix_el, f'{{{SBOL3}}}displayId')
        ix_display.text = ix_name

        # Interaction type
        sbo_type = SBO_INTERACTION_TYPES.get(ix['type'], 'SBO:0000231')
        ix_type = ET.SubElement(ix_el, f'{{{SBOL3}}}type')
        ix_type.set(f'{{{RDF}}}resource', sbo_type)

        # Participation: from
        if ix['from'] in comp_uri_map:
            part_from = ET.SubElement(ix_el, f'{{{SBOL3}}}hasParticipation')
            p_from = ET.SubElement(part_from, f'{{{SBOL3}}}Participation')
            p_from.set(f'{{{RDF}}}about', f"{ix_uri}/from")

            p_from_role = ET.SubElement(p_from, f'{{{SBOL3}}}role')
            role_key = SBO_ROLES.get(ix['type'], {}).get('from', 'SBO:0000003')
            p_from_role.set(f'{{{RDF}}}resource', role_key)

            p_from_participant = ET.SubElement(p_from, f'{{{SBOL3}}}participant')
            p_from_participant.set(f'{{{RDF}}}resource', comp_uri_map[ix['from']])

        # Participation: to
        if ix['to'] in comp_uri_map:
            part_to = ET.SubElement(ix_el, f'{{{SBOL3}}}hasParticipation')
            p_to = ET.SubElement(part_to, f'{{{SBOL3}}}Participation')
            p_to.set(f'{{{RDF}}}about', f"{ix_uri}/to")

            p_to_role = ET.SubElement(p_to, f'{{{SBOL3}}}role')
            role_key = SBO_ROLES.get(ix['type'], {}).get('to', 'SBO:0000003')
            p_to_role.set(f'{{{RDF}}}resource', role_key)

            p_to_participant = ET.SubElement(p_to, f'{{{SBOL3}}}participant')
            p_to_participant.set(f'{{{RDF}}}resource', comp_uri_map[ix['to']])

    # --- Part Definitions (top-level Components for each part) ---
    for comp in circuit['components']:
        comp_name = sanitize_uri(comp['name'])
        part_uri = f"{BASE_URI}parts/{comp_name}"

        part_def = ET.SubElement(root, f'{{{SBOL3}}}Component')
        part_def.set(f'{{{RDF}}}about', part_uri)

        pd_display = ET.SubElement(part_def, f'{{{SBOL3}}}displayId')
        pd_display.text = comp_name

        pd_name = ET.SubElement(part_def, f'{{{DCTERMS}}}title')
        pd_name.text = comp['name']

        # Namespace (required by SBOL3 spec)
        pd_ns = ET.SubElement(part_def, f'{{{SBOL3}}}hasNamespace')
        pd_ns.set(f'{{{RDF}}}resource', BASE_URI.rstrip('/'))

        # Type: DNA
        pd_type = ET.SubElement(part_def, f'{{{SBOL3}}}type')
        pd_type.set(f'{{{RDF}}}resource', 'http://www.biopax.org/release/biopax-level3.owl#DnaRegion')

        # Role
        so_role = SO_ROLES.get(comp['type'], SO_ROLES['other'])
        pd_role = ET.SubElement(part_def, f'{{{SBOL3}}}role')
        pd_role.set(f'{{{RDF}}}resource', so_role)

    # Serialize to string
    rough = ET.tostring(root, encoding='unicode', xml_declaration=False)
    dom = minidom.parseString(rough)
    pretty = dom.toprettyxml(indent="  ", encoding=None)

    # Remove extra blank lines from minidom
    lines = [l for l in pretty.split('\n') if l.strip()]
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description="Convert Newgenes circuit JSON to SBOL3 XML")
    parser.add_argument("input", nargs="?", help="JSON string or file")
    parser.add_argument("-o", "--output", help="Output XML file")
    parser.add_argument("--jsonl", help="JSONL training file")
    parser.add_argument("--index", type=int, default=0, help="Example index in JSONL")
    parser.add_argument("--name", help="Circuit name")
    args = parser.parse_args()

    if args.jsonl:
        with open(args.jsonl) as f:
            lines = f.readlines()
        d = json.loads(lines[args.index])
        circuit = json.loads(d['messages'][2]['content'])
        description = d['messages'][1]['content']
    elif args.input:
        if args.input.endswith('.json'):
            with open(args.input) as f:
                circuit = json.load(f)
        else:
            circuit = json.loads(args.input)
        description = None
    else:
        circuit = json.load(sys.stdin)
        description = None

    xml = circuit_to_sbol3(circuit, circuit_name=args.name, description=description)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(xml)
        print(f"Written to {args.output} ({len(xml)} bytes)")
    else:
        print(xml)


if __name__ == "__main__":
    main()
