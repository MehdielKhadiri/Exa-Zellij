import importlib.util
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location("configure",Path(__file__).resolve().parents[1]/"configure.py")
configure=importlib.util.module_from_spec(spec)
spec.loader.exec_module(configure)


class DetectionTests(unittest.TestCase):
    def test_toolchains(self):
        for tools, expected in (({},"cpu"),({"nvcc":"nvcc"},"cuda"),({"hipcc":"hipcc"},"hip")):
            self.assertEqual(configure.detect_backend([],tools.get)[0],expected)
        with self.assertRaisesRegex(ValueError,"Both CUDA and HIP"):
            configure.detect_backend([],{"nvcc":"nvcc","hipcc":"hipcc"}.get)

    def test_installed_kokkos_takes_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            config=Path(directory)/"lib/cmake/Kokkos/KokkosConfigCommon.cmake"
            config.parent.mkdir(parents=True)
            for device,backend in (("SERIAL","cpu"),("CUDA;SERIAL","cuda"),("HIP;SERIAL","hip")):
                config.write_text(f'SET(Kokkos_DEVICES {device})\n')
                self.assertEqual(configure.detect_backend([directory],lambda tool:"present")[0],backend)


if __name__=="__main__":
    unittest.main()
