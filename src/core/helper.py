def format_spacy_entities(spacy_ents):
    """Helper to format spaCy entity objects for Firestore storage."""
    return [{'text': ent.text, 'label': ent.label_} for ent in spacy_ents]

def format_nl_api_entities(nl_api_ents):
    """Helper to format NL API entity objects for Firestore storage."""
    return [
        {
            "name": entity.name,
            "type": entity.type_.name,
            "salience": entity.salience,
            "wikipedia_url": entity.metadata.get("wikipedia_url", "")
        }
        for entity in nl_api_ents
    ]