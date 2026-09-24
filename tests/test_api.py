import math
import os
import json
import tempfile
from pathlib import Path
import subprocess
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from fqo import Config, CommandObjective, minimize
from fqo.api import executable_path


class ConfigTests(unittest.TestCase):
    def test_rejects_typo_and_invalid_choices(self):
        for cfg in (Config(family="fob"), Config(budget=0), Config(budget=True),
                    Config(components={"surrogat": "rbf"}),
                    Config(components={"surrogate": "typo"}),
                    Config(family="fbd", components={"surrogate": "rbf"})):
            with self.subTest(config=cfg), self.assertRaises(ValueError):
                cfg.arguments(3)


class NativeTests(unittest.TestCase):
    def test_native_analytic_result_is_consistent(self):
        for family in ("fbd", "fbo"):
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory)/"result.json"
                extra = ["--n-init","20"] if family=="fbo" else []
                run = subprocess.run([executable_path(),"4","3","80","1","shiftedsphere",
                    "--optimizer-family",family,"--max-depth","2","--fbo-doe","halton",
                    "--seed","7","--result-json",str(output),*extra],cwd=directory,capture_output=True,text=True,timeout=30)
                self.assertEqual(run.returncode,0,run.stdout+run.stderr)
                result=json.loads(output.read_text())
                self.assertAlmostEqual(result['fun'],5+sum((x-1)**2 for x in result['x']),places=9)

    def test_both_families_use_the_user_objective(self):
        for family in ("fbd", "fbo"):
            seen = []
            def objective(x):
                self.assertTrue(-3.000000001 <= x[0] <= -0.999999999)
                self.assertTrue(99.999999999 <= x[1] <= 120.000000001)
                seen.append(list(x))
                return 7 + (x[0]+1.37)**2 + ((x[1]-107.3)/10)**2
            result = minimize(objective, [(-3,-1),(100,120)],
                              config=Config(family=family, budget=45 if family=="fbo" else None, max_depth=2))
            with self.subTest(family=family):
                self.assertGreater(result.nfev, 5)
                if family=="fbo": self.assertLessEqual(result.nfev, 45)
                if family=="fbo": self.assertIn("[DEPTH",result.log)
                self.assertEqual(result.nfev, len(seen))
                expected = min(7+(x[0]+1.37)**2+((x[1]-107.3)/10)**2 for x in seen)
                self.assertEqual(result.fun, expected)
                self.assertIn(result.x, seen)

    def test_components(self):
        for surrogate in ("pca-gp", "idw", "rbf", "none"):
            result = minimize(lambda x: 3+sum((v-.23)**2 for v in x), [(0,1)]*3,
                config=Config(budget=40, max_depth=2, components={"surrogate":surrogate,"decomposition":"trisection"}))
            self.assertTrue(math.isfinite(result.fun), surrogate)

    def test_fbd_geometries(self):
        for geometry in ("hypersphere","hypercube","bisection","trisection","voronoi","simplex"):
            result = minimize(lambda x: sum((v-.23)**2 for v in x), [(0,1)]*2,
                config=Config(family="fbd",max_depth=2,components={"decomposition":geometry}))
            self.assertGreater(result.nfev,1,geometry)

    def test_batch_and_scalar_agree(self):
        f = lambda x: sum((v-.27)**2 for v in x)
        cfg = Config(budget=35,max_depth=2,components={"surrogate":"idw"})
        a = minimize(f,[(0,1)]*2,config=cfg)
        b = minimize(f,[(0,1)]*2,config=cfg,batch_objective=lambda rows:[f(x) for x in rows])
        self.assertEqual((a.x,a.fun,a.nfev),(b.x,b.fun,b.nfev))

    def test_callable_failure_is_not_a_penalty(self):
        def broken(x):
            raise LookupError("simulation unavailable")
        with self.assertRaisesRegex(LookupError,"simulation unavailable"):
            minimize(broken,[(0,1)]*2)
        with self.assertRaisesRegex(ValueError,"non-finite"):
            minimize(lambda x:float("nan"),[(0,1)]*2)
        with self.assertRaisesRegex(ValueError,"one value per row"):
            minimize(lambda x:0,[(0,1)]*2,batch_objective=lambda rows:[])

    def test_external_executable(self):
        callback = CommandObjective([sys.executable,"-c",
            'import json,sys; x=json.load(sys.stdin)["x"]; print(sum(v*v for v in x))'])
        result = minimize(callback,[(-1,1)]*2,config=Config(family="fbo",budget=8))
        self.assertLessEqual(result.nfev,8)

    def test_unknown_native_objective_fails(self):
        run = subprocess.run([executable_path(),"4","2","30","1","sphere_typo"],capture_output=True,text=True)
        self.assertNotEqual(run.returncode,0)
        self.assertIn("Unknown objective",run.stderr)


if __name__ == "__main__":
    unittest.main()
