from edgar_rag.ingest import chunk_section, split_sections

def test_chunks_stay_under_target_plus_overlap():
    body = "\n\n".join(f"Paragraph {i}. " + "word " * 60 for i in range(20))
    chunks = chunk_section(body, target=1200, overlap=150)
    assert chunks
    assert all(len(c) < 1200 + 400 for c in chunks)


def test_sections_split_on_item_boundaries():
    text = (
        "Item 1A. Risk Factors\n\n" + "risk " * 200 + "\n\n"
        "Item 7. Management Discussion and Analysis\n\n" + "mda " * 200
    )
    sections = split_sections(text)
    items = {s[0] for s in sections}
    assert "1A" in items and "7" in items
    # bodies must not bleed across the boundary
    risk_body = next(s[2] for s in sections if s[0] == "1A")
    assert "mda" not in risk_body


def test_toc_entries_are_dropped():
    text = "Item 1A. Risk Factors ... 12\n\nItem 7. MD&A ... 34\n"
    assert split_sections(text) == []
