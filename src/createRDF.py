from rdflib import Graph, Literal, RDF, URIRef, Namespace #basic RDF handling
from rdflib.namespace import RDF, RDFS, TIME, XSD, OWL
import pandas as pd
from pathlib import Path

main_path = Path().absolute().parent
data_path = main_path / 'data'

# full
df = pd.read_csv(data_path / 'AccountRDF.csv', encoding='utf-8')
ns_acc = Namespace('http://fsqa.com/acc#')
namespace_dict = {
    'acc': ns_acc, 'time': TIME, 'rdf': RDF, 'rdfs': RDFS, 'owl': OWL, None: ''
}
g = Graph()
g.bind('rdf', RDF)
g.bind('time', TIME)
g.bind('owl', OWL)
g.bind('acc', ns_acc, override=True)
for index, row in df.iterrows():
    s_ns, s = row['subject'].split(':')
    p_ns, p = row['predicate'].split(':')
    if len(row['object'].split(':')) < 2:
        o_ns, o = None, row['object']
    else:
        o_ns, o = row['object'].split(':')
    
    if o_ns is None:
        g.add( (URIRef(namespace_dict[s_ns]+s), URIRef(namespace_dict[p_ns]+p), Literal(o, datatype=XSD.string)) )
    else:
        g.add( (URIRef(namespace_dict[s_ns]+s), URIRef(namespace_dict[p_ns]+p), URIRef(namespace_dict[o_ns]+o)) )

g.serialize(data_path / 'AccountRDF.ttl', format='turtle')
g.serialize(data_path / 'AccountRDF.xml', format='xml', encoding='utf-8')
print("Done!")