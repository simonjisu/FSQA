from rdflib import Literal, URIRef

def convert_to_string(x):
    if isinstance(x, URIRef):
        if len(x.split('#')) == 2:
            return x.split('#')[1]
        else:
            raise ValueError(f'Split error {x}')
    elif isinstance(x, Literal):
        return str(x)
    else:
        raise ValueError(f'Returned None')