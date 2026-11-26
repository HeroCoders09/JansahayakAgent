import unittest

from jansahayak_agent import (
    build_fallback_queries,
    is_official_indian_gov_url,
    normalise_url,
    parse_json_object,
    replace_source_ids,
    sanitise_answer_links,
    search_official_web,
)


class JanSahayakDynamicAgentTests(unittest.TestCase):
    def test_official_domain_allowlist(self):
        self.assertTrue(is_official_indian_gov_url("https://it.uk.gov.in/cm-helpline/"))
        self.assertTrue(is_official_indian_gov_url("https://dehradun.nic.in/service/"))
        self.assertTrue(is_official_indian_gov_url("https://gov.in/"))
        self.assertFalse(is_official_indian_gov_url("http://it.uk.gov.in/"))
        self.assertFalse(is_official_indian_gov_url("https://uk.gov.in.example.com/"))
        self.assertFalse(is_official_indian_gov_url("https://example.com/uk.gov.in"))

    def test_fallback_queries_do_not_require_catalogue(self):
        queries = build_fallback_queries("municipal garbage complaint", "Roorkee")
        self.assertEqual(len(queries), 3)
        self.assertTrue(any("site:gov.in" in query for query in queries))
        self.assertTrue(any("Roorkee" in query for query in queries))

    def test_parse_planner_json(self):
        parsed = parse_json_object('```json\n{"intent":"complaint","search_queries":["site:gov.in complaint"]}\n```')
        self.assertEqual(parsed["intent"], "complaint")

    def test_search_filters_non_official_domains(self):
        def fake_search(query, max_results):
            return [
                {"title": "Official", "url": "https://it.uk.gov.in/cm-helpline/", "snippet": "Official info"},
                {"title": "Blog", "url": "https://example.com/how-to", "snippet": "Blog info"},
            ]

        results, queries, errors = search_official_web(["complaint Uttarakhand"], search_fn=fake_search)
        self.assertEqual(len(results), 1)
        self.assertEqual(len(queries), 1)
        self.assertEqual(errors, [])

    def test_source_id_replacement(self):
        sources = [{"source_id": "S1", "title": "CM Helpline", "url": "https://cmhelpline.uk.gov.in/"}]
        answer = replace_source_ids("Use the official route [S1].", sources)
        self.assertIn("[CM Helpline](https://cmhelpline.uk.gov.in/)", answer)


    def test_dynamic_search_accepts_unlisted_service_and_uses_eight_queries(self):
        queries = [f"unlisted public service {i}" for i in range(10)]
        seen = []

        def fake_search(query, max_results):
            seen.append(query)
            return [
                {"title": "Official", "url": f"https://service{i}.gov.in/" if (i:=len(seen)) else "https://gov.in/", "snippet": "Official"}
            ]

        results, executed, errors = search_official_web(queries, search_fn=fake_search)
        self.assertEqual(len(executed), 8)
        self.assertEqual(len(seen), 8)
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 8)

    def test_normalise_url(self):
        self.assertEqual(normalise_url("https://example.gov.in/path/#section"), "https://example.gov.in/path")

    def test_sanitise_answer_links_only_keeps_discovered_urls(self):
        sources = [{"source_id": "S1", "title": "CM Helpline", "url": "https://cmhelpline.uk.gov.in/"}]
        answer = "[CM Helpline](https://cmhelpline.uk.gov.in/) [Fake](https://fake.example/)"
        cleaned = sanitise_answer_links(answer, sources)
        self.assertIn("[CM Helpline](https://cmhelpline.uk.gov.in/)", cleaned)
        self.assertIn("link not verified by JanSahayak", cleaned)
        raw_url = sanitise_answer_links("Unverified: https://phishing.example/apply", sources)
        self.assertIn("unverified URL omitted", raw_url)


if __name__ == "__main__":
    unittest.main(verbosity=2)
