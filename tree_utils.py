

def get_head(t, root='.'):
    if t.dep_ == 'ROOT':
        return [t.text] + [root]
    else:
        r = [t.text] + get_head(t.head)
        return r


def get_trees(doc, min_size=4, max_size=8):
    result = list()
    for t in doc:
        a = get_head(t)
        b = [c.text for c in t.children]
        if len(a) >= min_size and len(a) < max_size:
            result.append(a)
    return result


def get_flattened(doc, min_size=4, max_size=6, fill='.'):
    result = ''
    trees = get_trees(doc, min_size=min_size, max_size=max_size)
    if len(trees) > 0:
        result = ''
        for t in trees:
            result += " " + " ".join([s for s in t] + (max_size - len(t)) * [fill])
        result = result.strip()
    return result


if __name__ == '__main__':
    import spacy
    nlp = spacy.load('en_core_web_lg')
    t = 'The best is a text about a flattened document.'
    print get_flattened(t, nlp)
