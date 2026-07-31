import unittest

from videodub.tts import _atempo_chain


class TtsTests(unittest.TestCase):
    def test_atempo_chain_splits_large_factor(self) -> None:
        value = _atempo_chain(5.0)
        self.assertTrue(value.startswith("atempo=2.0,atempo=2.0,"))
        self.assertIn("atempo=1.250000", value)

if __name__ == "__main__":
    unittest.main()
