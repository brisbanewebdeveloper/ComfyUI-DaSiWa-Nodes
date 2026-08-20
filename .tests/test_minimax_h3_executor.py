import importlib
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


ROOT = Path(__file__).parents[1]
PACKAGE_NAME = "dasiwa_executor_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)
executor = importlib.import_module(
    f"{PACKAGE_NAME}.nodes.nodes_minimax_h3_executor"
)


class MiniMaxH3DirectorExecutorTests(unittest.TestCase):
    def test_single_stage_sample_and_decode_contract(self):
        latent = {
            "samples": torch.ones(1),
            "downscale_ratio_spacial": 8,
            "downscale_ratio_temporal": 4,
        }
        images = torch.zeros(5, 8, 8, 3)
        video_latent = {"samples": torch.ones(1)}
        audio_latent = {"samples": torch.ones(1)}
        audio = {"waveform": torch.ones(1, 1, 16), "sample_rate": 44100}

        sigma_shift = mock.Mock()
        sigma_shift.execute.return_value = ("shifted-model",)
        guide_adapter = mock.Mock()
        guide_adapter.apply.return_value = ("positive", latent)
        vae_decode = mock.Mock()
        vae_decode.decode.return_value = (images,)

        with (
            mock.patch.object(
                executor,
                "normalize_guide",
                return_value=SimpleNamespace(
                    mode="FL2VA", width=1344, height=768
                ),
            ),
            mock.patch.object(
                executor, "MiniMaxH3DirectorGuide", return_value=guide_adapter
            ),
            mock.patch.object(
                executor, "_native_node", return_value=sigma_shift
            ),
            mock.patch.object(
                executor.comfy.sample,
                "fix_empty_latent_channels",
                return_value=latent["samples"],
            ),
            mock.patch.object(
                executor.comfy.sample,
                "prepare_noise",
                return_value=torch.zeros(1),
            ),
            mock.patch.object(
                executor.comfy.sample,
                "sample",
                return_value=torch.full((1,), 2.0),
            ) as sample,
            mock.patch.object(
                executor.latent_preview, "prepare_callback", return_value=None
            ),
            mock.patch.object(
                executor.LTXVSeparateAVLatent,
                "execute",
                return_value=(video_latent, audio_latent),
            ),
            mock.patch.object(executor, "VAEDecode", return_value=vae_decode),
            mock.patch.object(
                executor.VAEDecodeAudio, "execute", return_value=(audio,)
            ),
        ):
            result = executor.MiniMaxH3DirectorExecutor().execute(
                "model", "clip", "video-vae", {"mode": "FL2VA"},
                7, 25, 1.0, "res_multistep", "simple",
                12.0, 3.0, 24.0, "audio-vae",
            )

        self.assertIs(result[0], images)
        self.assertIs(result[1], audio)
        self.assertEqual(result[2:4], (24.0, 5))
        self.assertNotIn("downscale_ratio_spacial", result[4])
        self.assertNotIn("downscale_ratio_temporal", result[4])
        self.assertEqual(sample.call_args.args[7], [])
        self.assertEqual(sample.call_args.kwargs["seed"], 7)
        self.assertIn("FL2VA: 1344x768, 5 frames", result[5])


if __name__ == "__main__":
    unittest.main()
