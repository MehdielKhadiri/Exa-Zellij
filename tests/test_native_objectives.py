"""Catch row/column layout mistakes and native point/value mismatches."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--executable",required=True)
    args=p.parse_args()
    shifted=[0.35+0.9*((37*j+11)%101)/100 for j in range(3)]
    targets={"shifted-ellipsoid":shifted,"shifted-rastrigin":shifted,
             "rosenbrock":[1.0]*3,"schwefel":[420.9687462275036]*3}
    with tempfile.TemporaryDirectory(prefix="fqo-native-layout-") as temporary:
        root=Path(temporary)
        for objective,target in targets.items():
            doe=root/(objective+".csv")
            # Seven rows, three columns, distinct entries: reinterpretation as
            # column-major memory must not silently evaluate different points.
            doe.write_text("\n".join(",".join(format(x+delta,".17g") for x in target)
                                     for delta in (.01,.02,0,.03,-.01,-.02,.04))+"\n")
            result=root/(objective+".json")
            lo,hi=(-500,500) if objective=="schwefel" else (-5,5)
            command=[args.executable,"4","3","80","0",objective,
                     "--optimizer-family","fbo","--fbo-decomposition","trisection",
                     "--fbo-split-policy","longest-side","--fbo-doe","file","--doe-file",str(doe),
                     "--n-init","7","--fbo-surrogate","pca-gp","--fbo-exploitation","best-cell",
                     "--fbo-budget","fixed","--doe-frac","0.25","--explore-frac","0.65",
                     "--exploit-frac","0.10","--max-depth","2","--seed","0",
                     "--lower-bound",str(lo),"--upper-bound",str(hi),"--result-json",str(result)]
            run=subprocess.run(command,cwd=root,capture_output=True,text=True,timeout=45)
            assert run.returncode==0,run.stdout+run.stderr
            record=json.loads(result.read_text())
            assert abs(record["fun"])<1e-7,record
            assert all(abs(x-y)<1e-7 for x,y in zip(record["x"],target)),record
            assert record["host_evaluations"]==0 and record["device_evaluations"]>0,record
            assert record["evaluations"]==sum(record["rank_evaluations"]),record
            print("PASS",objective,"known optimum, native batch layout, counters")


if __name__=="__main__":
    main()
