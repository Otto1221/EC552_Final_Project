#!/usr/bin/env python3
"""JSON → SBOL2 RDF/XML converter — parallel to json_to_sbol3.py, emits SBOL2.

iBioSim and other legacy SBOL2 tools (libSBOLj 2.x) consume this directly.
Compliant URI scheme: {namespace}/{displayId}/{version} with proper persistentIdentity chains.

Usage:
    python json_to_sbol2.py results/demo_last.txt -o results/demo_last_sbol2.xml
"""
import json, re, sys, argparse
from xml.etree import ElementTree as ET
from xml.dom import minidom

# Namespaces
SBOL2   = "http://sbols.org/v2#"
RDF     = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
DCTERMS = "http://purl.org/dc/terms/"
BIOPAX  = "http://www.biopax.org/release/biopax-level3.owl#"
SO_URI  = "http://identifiers.org/so/"
SBO_URI = "http://identifiers.org/biomodels.sbo/"

SO_ROLES = {
    'promoter':   'SO:0000167',
    'rbs':        'SO:0000139',
    'cds':        'SO:0000316',
    'terminator': 'SO:0000141',
    'operator':   'SO:0000057',
    'other':      'SO:0000110',
}

SBO_INTERACTION_TYPES = {
    'activation':        'SBO:0000170',
    'repression':        'SBO:0000169',
    'inhibition':        'SBO:0000169',
    'transcription':     'SBO:0000589',
    'translation':       'SBO:0000184',
    'production':        'SBO:0000393',
    'complex_formation': 'SBO:0000177',
    'degradation':       'SBO:0000179',
}

SBO_ROLES = {
    'activation':        {'from': 'SBO:0000459', 'to': 'SBO:0000643'},
    'repression':        {'from': 'SBO:0000020', 'to': 'SBO:0000642'},
    'inhibition':        {'from': 'SBO:0000020', 'to': 'SBO:0000642'},
    'transcription':     {'from': 'SBO:0000645', 'to': 'SBO:0000011'},
    'translation':       {'from': 'SBO:0000645', 'to': 'SBO:0000011'},
    'production':        {'from': 'SBO:0000010', 'to': 'SBO:0000011'},
    'complex_formation': {'from': 'SBO:0000280', 'to': 'SBO:0000280'},
    'degradation':       {'from': 'SBO:0000010', 'to': 'SBO:0000011'},
}

BASE_URI = "https://newgenes.org"
VERSION = "1"


def sanitize(name):
    """Valid SBOL2 displayId: alphanumeric + underscore, must start with letter or underscore."""
    if not name:
        return "unnamed"
    s = re.sub(r'[^a-zA-Z0-9_]', '_', str(name))
    if s and s[0].isdigit():
        s = '_' + s
    return s or "unnamed"


def _add_identity(el, persistent_id, display_id):
    """Add the three standard identity properties to an SBOL2 element."""
    pid = ET.SubElement(el, f'{{{SBOL2}}}persistentIdentity')
    pid.set(f'{{{RDF}}}resource', persistent_id)
    disp = ET.SubElement(el, f'{{{SBOL2}}}displayId')
    disp.text = display_id
    ver = ET.SubElement(el, f'{{{SBOL2}}}version')
    ver.text = VERSION


def circuit_to_sbol2(circuit, circuit_name=None):
    """Convert circuit JSON to SBOL2 RDF/XML.

    Structure:
      - One ComponentDefinition per part (with type=DnaRegion and SO role).
      - One ModuleDefinition for the circuit, containing:
          - FunctionalComponent per part (references CD via definition).
          - Interaction per JSON interaction, with Participations
            referencing FunctionalComponents.
    """
    if not circuit_name:
        cds_names = [c['name'] for c in circuit.get('components', []) if c.get('type') == 'cds']
        circuit_name = '_'.join(cds_names[:3]) + '_circuit' if cds_names else 'circuit'
    md_name = sanitize(circuit_name) + "_mod"

    ET.register_namespace('sbol',    SBOL2)
    ET.register_namespace('rdf',     RDF)
    ET.register_namespace('dcterms', DCTERMS)

    root = ET.Element(f'{{{RDF}}}RDF')

    # --- ComponentDefinitions (one per part) ---
    cd_uri_map = {}  # JSON part name → CD uri
    for comp in circuit.get('components', []):
        if not isinstance(comp, dict) or not comp.get('name'):
            continue
        cname = sanitize(comp['name'])
        cd_persist = f"{BASE_URI}/{cname}"
        cd_uri     = f"{cd_persist}/{VERSION}"
        cd_uri_map[comp['name']] = cd_uri

        cd = ET.SubElement(root, f'{{{SBOL2}}}ComponentDefinition')
        cd.set(f'{{{RDF}}}about', cd_uri)
        _add_identity(cd, cd_persist, cname)

        title = ET.SubElement(cd, f'{{{DCTERMS}}}title')
        title.text = comp['name']
        if comp.get('description'):
            desc = ET.SubElement(cd, f'{{{DCTERMS}}}description')
            desc.text = comp['description']

        type_el = ET.SubElement(cd, f'{{{SBOL2}}}type')
        type_el.set(f'{{{RDF}}}resource', f'{BIOPAX}DnaRegion')

        so = SO_ROLES.get(comp.get('type', 'other'), SO_ROLES['other'])
        role = ET.SubElement(cd, f'{{{SBOL2}}}role')
        role.set(f'{{{RDF}}}resource', f'{SO_URI}{so}')

    # --- ModuleDefinition ---
    md_persist = f"{BASE_URI}/{md_name}"
    md_uri     = f"{md_persist}/{VERSION}"

    md = ET.SubElement(root, f'{{{SBOL2}}}ModuleDefinition')
    md.set(f'{{{RDF}}}about', md_uri)
    _add_identity(md, md_persist, md_name)

    title = ET.SubElement(md, f'{{{DCTERMS}}}title')
    title.text = md_name.replace('_', ' ')

    # FunctionalComponents
    fc_uri_map = {}  # JSON part name → FC uri
    for comp in circuit.get('components', []):
        if not isinstance(comp, dict) or not comp.get('name'):
            continue
        cname = sanitize(comp['name'])
        fc_persist = f"{md_persist}/fc_{cname}"
        fc_uri     = f"{fc_persist}/{VERSION}"
        fc_uri_map[comp['name']] = fc_uri

        wrapper = ET.SubElement(md, f'{{{SBOL2}}}functionalComponent')
        fc = ET.SubElement(wrapper, f'{{{SBOL2}}}FunctionalComponent')
        fc.set(f'{{{RDF}}}about', fc_uri)
        _add_identity(fc, fc_persist, f"fc_{cname}")

        access = ET.SubElement(fc, f'{{{SBOL2}}}access')
        access.set(f'{{{RDF}}}resource', f'{SBOL2}public')
        direction = ET.SubElement(fc, f'{{{SBOL2}}}direction')
        direction.set(f'{{{RDF}}}resource', f'{SBOL2}inout')

        defn = ET.SubElement(fc, f'{{{SBOL2}}}definition')
        defn.set(f'{{{RDF}}}resource', cd_uri_map[comp['name']])

    # Interactions
    for i, ix in enumerate(circuit.get('interactions', []) or []):
        if not isinstance(ix, dict):
            continue
        ix_name    = f"interaction_{i}"
        ix_persist = f"{md_persist}/{ix_name}"
        ix_uri     = f"{ix_persist}/{VERSION}"

        wrapper = ET.SubElement(md, f'{{{SBOL2}}}interaction')
        ix_el = ET.SubElement(wrapper, f'{{{SBOL2}}}Interaction')
        ix_el.set(f'{{{RDF}}}about', ix_uri)
        _add_identity(ix_el, ix_persist, ix_name)

        sbo = SBO_INTERACTION_TYPES.get(ix.get('type', ''), 'SBO:0000231')
        tel = ET.SubElement(ix_el, f'{{{SBOL2}}}type')
        tel.set(f'{{{RDF}}}resource', f'{SBO_URI}{sbo}')

        # Participations
        for direction_key in ('from', 'to'):
            v = ix.get(direction_key)
            if not v or v not in fc_uri_map:
                continue
            part_name    = f"participation_{direction_key}"
            part_persist = f"{ix_persist}/{part_name}"
            part_uri     = f"{part_persist}/{VERSION}"

            p_wrapper = ET.SubElement(ix_el, f'{{{SBOL2}}}participation')
            part = ET.SubElement(p_wrapper, f'{{{SBOL2}}}Participation')
            part.set(f'{{{RDF}}}about', part_uri)
            _add_identity(part, part_persist, part_name)

            sbo_role = SBO_ROLES.get(ix.get('type', ''), {}).get(direction_key, 'SBO:0000003')
            role = ET.SubElement(part, f'{{{SBOL2}}}role')
            role.set(f'{{{RDF}}}resource', f'{SBO_URI}{sbo_role}')

            participant = ET.SubElement(part, f'{{{SBOL2}}}participant')
            participant.set(f'{{{RDF}}}resource', fc_uri_map[v])

    # Serialize + pretty-print
    rough = ET.tostring(root, encoding='unicode')
    dom = minidom.parseString(rough)
    pretty = dom.toprettyxml(indent='  ')
    lines = [l for l in pretty.split('\n') if l.strip()]
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description="Convert circuit JSON to SBOL2 RDF/XML (iBioSim compatible)")
    parser.add_argument("input", help="JSON file or '-' for stdin")
    parser.add_argument("-o", "--output", required=True, help="Output SBOL2 XML file")
    parser.add_argument("--name", help="Circuit name (overrides JSON 'name' field)")
    args = parser.parse_args()

    data = sys.stdin.read() if args.input == '-' else open(args.input).read()

    try:
        circuit = json.loads(data)
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', data, re.DOTALL)
        if not m:
            sys.exit("could not parse JSON from input")
        circuit = json.loads(m.group())

    xml = circuit_to_sbol2(circuit, circuit_name=args.name or circuit.get('name'))
    with open(args.output, 'w') as f:
        f.write(xml)
    print(f"wrote {args.output} ({len(xml):,} bytes)")


if __name__ == "__main__":
    main()
