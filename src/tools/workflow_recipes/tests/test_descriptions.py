"""Unit tests for index.json (catalog) handling: parsing, category aggregation,
and the description-field policy."""

import json
import os
import tempfile
import unittest

from src.tools.workflow_recipes import parser as P
from src.tools.workflow_recipes import recipe as R
from src.tools.workflow_recipes.recipe import RecipeBuilder

OFFICIAL_INDEX = [
    {
        "moduleName": "default",
        "title": "Image Tools",
        "blueprints": [
            {"name": "sharpen", "title": "Sharpen",
             "description": "Sharpens an image using a GPU shader.", "mediaType": "image"},
            {"name": "image_blur", "title": "Image Blur",
             "description": "Blurs an image.", "mediaType": "image"},
        ],
    },
    {
        "moduleName": "default",
        "title": "Video Tools",
        "blueprints": [
            {"name": "video_stitch", "title": "Video Stitch",
             "description": "Stitches videos.", "mediaType": "video"},
        ],
    },
]

# Custom-shape index (no descriptions) must be handled gracefully.
CUSTOM_INDEX = [{"templates": [{"name": "my_custom_wf", "models": [], "io": {}}]}]


def _leaf(graphs):
    """The single (task, model) leaf for a homogeneous set of graphs."""
    return RecipeBuilder().build(graphs).leaves[0]


class TestLoadDescriptions(unittest.TestCase):
    def test_parses_official_and_custom_shapes(self):
        with tempfile.TemporaryDirectory() as d:
            off = os.path.join(d, "official")
            cust = os.path.join(d, "custom", "templates")
            os.makedirs(off)
            os.makedirs(cust)
            with open(os.path.join(off, "index.json"), "w", encoding="utf-8") as f:
                json.dump(OFFICIAL_INDEX, f)
            with open(os.path.join(cust, "index.json"), "w", encoding="utf-8") as f:
                json.dump(CUSTOM_INDEX, f)
            meta = P.load_descriptions(
                {"official": off, "custom": os.path.join(d, "custom")}, log=lambda *a: None
            )
        self.assertEqual(meta["sharpen"]["category"], "Image Tools")
        self.assertEqual(meta["sharpen"]["description"], "Sharpens an image using a GPU shader.")
        self.assertEqual(meta["video_stitch"]["category"], "Video Tools")
        # Custom entry recorded but with no description.
        self.assertIn("my_custom_wf", meta)
        self.assertIsNone(meta["my_custom_wf"]["description"])

    def test_missing_index_is_not_fatal(self):
        with tempfile.TemporaryDirectory() as d:
            meta = P.load_descriptions({"official": d}, log=lambda *a: None)
        self.assertEqual(meta, {})


def _graph(name, classes, source="official", category=None, description=None):
    nodes = [{"id": i, "type": c, "widgets_values": []} for i, c in enumerate(classes)]
    g = P.parse_ui({"nodes": nodes, "links": []}, name, name + ".json", source)
    g.category = category
    g.index_description = description
    g.index_title = name.title()
    return g


class TestCategoryAggregation(unittest.TestCase):
    def test_pure_category(self):
        gs = [_graph("a", ["KSampler"], category="Image Tools"),
              _graph("b", ["KSampler"], category="Image Tools")]
        info = R._category_info(gs)
        self.assertEqual(info["primary"], "Image Tools")
        self.assertTrue(info["pure"])

    def test_mixed_category_not_pure(self):
        gs = [_graph("a", ["KSampler"], category="Image Tools"),
              _graph("b", ["KSampler"], category="Video Tools"),
              _graph("c", ["KSampler"], category="Image Tools")]
        info = R._category_info(gs)
        self.assertEqual(info["primary"], "Image Tools")   # 2 vs 1
        self.assertFalse(info["pure"])
        self.assertEqual(info["distribution"], {"Image Tools": 2, "Video Tools": 1})


class TestDescriptionField(unittest.TestCase):
    def test_catalog_description_used_even_for_custom_nodes(self):
        # An authoritative catalog description is used directly as the type's
        # description (even for custom-node types), with provenance noted.
        g = _graph("vid", ["VHS_VideoCombine"], source="official",
                   category="Video Tools", description="Combines frames into a video.")
        recipe = _leaf([g])
        self.assertEqual(recipe["description"], "Combines frames into a video.")
        self.assertEqual(recipe["description_source"], "catalog")
        self.assertEqual(recipe["catalog_category"]["primary"], "Video Tools")
        self.assertEqual(recipe["member_descriptions"][0]["description"],
                         "Combines frames into a video.")

    def test_description_always_populated_for_custom(self):
        g = _graph("c", ["MyCustomThing"], source="custom")  # no category/desc
        g = P.enrich(g, {})  # MyCustomThing unresolved
        recipe = _leaf([g])
        self.assertTrue(recipe["description"])                 # never blank
        self.assertEqual(recipe["description_source"], "synthesized")

    def test_arrow_and_endash_sanitized_in_description(self):
        g = _graph("k", ["VHS_VideoCombine"], source="custom",
                   description="image → video – done")
        recipe = _leaf([g])
        self.assertNotIn("→", recipe["description"])
        self.assertNotIn("–", recipe["description"])
        self.assertIn("->", recipe["description"])


if __name__ == "__main__":
    unittest.main()
