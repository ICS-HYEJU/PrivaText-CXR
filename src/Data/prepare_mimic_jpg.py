"""Create audited MIMIC-CXR-JPG manifests for p10/p11 training and p12 testing."""
import argparse, hashlib, json, os
from pathlib import Path
import pandas as pd
from PIL import Image

LABEL_IDS = {"subject_id", "study_id"}

def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()

def readable(path):
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report_root", required=True)
    p.add_argument("--jpg_root", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_finding_ratio", type=float, default=2.0)
    p.add_argument("--verify_images", action="store_true")
    a = p.parse_args()
    rr, jr, out = Path(a.report_root).resolve(), Path(a.jpg_root).resolve(), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sp, cp, rp = rr/"mimic-cxr-2.0.0-split.csv", rr/"mimic-cxr-2.0.0-chexpert.csv", rr/"cxr-record-list.csv.gz"
    split = pd.read_csv(sp, dtype={"dicom_id": str})
    records = pd.read_csv(rp, dtype={"dicom_id": str})
    labels = pd.read_csv(cp)
    if split.dicom_id.duplicated().any() or records.dicom_id.duplicated().any():
        raise RuntimeError("dicom_id is not unique")
    df = split.merge(records, on=["dicom_id","subject_id","study_id"], how="left", validate="one_to_one")
    label_cols = [c for c in labels.columns if c not in LABEL_IDS]
    labels["_chexpert_found"] = True
    df = df.merge(labels, on=["subject_id","study_id"], how="left", validate="many_to_one")
    df["patient_prefix"] = "p" + df.subject_id.astype(str).str[:2]
    df = df[df.patient_prefix.isin(["p10","p11","p12"])].copy()
    pid, sid = "p"+df.subject_id.astype(str), "s"+df.study_id.astype(str)
    df["image_path"] = [str(jr/pre/pat/stu/f"{did}.jpg") for pre,pat,stu,did in zip(df.patient_prefix,pid,sid,df.dicom_id)]
    df["report_path"] = [str(rr/"files"/pre/pat/f"{stu}.txt") for pre,pat,stu in zip(df.patient_prefix,pid,sid)]
    df["image_exists"] = df.image_path.map(os.path.isfile)
    df["report_exists"] = df.report_path.map(os.path.isfile)
    if a.verify_images:
        vals = []
        for n, path in enumerate(df.image_path, 1):
            vals.append(readable(path) if os.path.isfile(path) else False)
            if n % 10000 == 0: print(f"[verify] {n}/{len(df)}")
        df["image_readable"] = vals
    else:
        df["image_readable"] = df.image_exists
    pos = df[label_cols].eq(1)
    df["chexpert_found"] = df["_chexpert_found"].fillna(False).astype(bool)
    df["positive_label_count"] = pos.sum(axis=1)
    df["positive_labels"] = pos.apply(lambda r: ";".join(r.index[r.to_numpy()]), axis=1)
    df["single_positive_label"] = pos.idxmax(axis=1).where(df.positive_label_count.eq(1), "")
    df["dataset_split"] = "excluded"
    p1011 = df.patient_prefix.isin(["p10","p11"])
    df.loc[p1011 & df.split.eq("train"), "dataset_split"] = "train"
    df.loc[p1011 & df.split.eq("validate"), "dataset_split"] = "validate"
    df.loc[p1011 & df.split.eq("test"), "dataset_split"] = "excluded_official_test"
    df.loc[df.patient_prefix.eq("p12"), "dataset_split"] = "test"
    groups = {k:set(v.subject_id) for k,v in df[df.dataset_split.isin(["train","validate","test"])].groupby("dataset_split")}
    overlap = {"train_validation":len(groups.get("train",set())&groups.get("validate",set())),
               "train_test":len(groups.get("train",set())&groups.get("test",set())),
               "validation_test":len(groups.get("validate",set())&groups.get("test",set()))}
    if any(overlap.values()): raise RuntimeError(f"patient leakage: {overlap}")
    df["exclusion_reason"] = ""
    df.loc[~df.image_exists, "exclusion_reason"] = "missing_image"
    df.loc[df.image_exists & ~df.image_readable, "exclusion_reason"] = "unreadable_image"
    df.loc[df.image_readable & ~df.report_exists, "exclusion_reason"] = "missing_report"
    valid = df.image_readable & df.report_exists
    before = df[valid & df.dataset_split.eq("train")].copy()
    before["balance_selected"] = True
    singles = before[before.positive_label_count.eq(1)]
    abnormal_counts = singles[singles.single_positive_label.ne("No Finding")].single_positive_label.value_counts()
    nf_idx = singles[singles.single_positive_label.eq("No Finding")].index
    cap = len(nf_idx)
    if a.no_finding_ratio >= 0 and len(abnormal_counts):
        cap = min(cap, int(abnormal_counts.max()*a.no_finding_ratio))
    if len(nf_idx) > cap:
        keep = set(before.loc[nf_idx].sample(cap, random_state=a.seed).index)
        drop = [i for i in nf_idx if i not in keep]
        before.loc[drop, "balance_selected"] = False
        df.loc[drop, "exclusion_reason"] = "no_finding_downsampled"
    train = before[before.balance_selected].copy()
    val = df[valid & df.dataset_split.eq("validate")].copy()
    test = df[valid & df.dataset_split.eq("test")].copy()
    excluded = df[(~valid)|df.dataset_split.eq("excluded_official_test")|df.exclusion_reason.ne("")].copy()
    cols = ["dicom_id","subject_id","study_id","patient_prefix","split","dataset_split","image_path","report_path",
            "image_exists","image_readable","report_exists","chexpert_found","positive_label_count",
            "positive_labels","single_positive_label","exclusion_reason"]
    df[cols].to_csv(out/"mimic_p10_p12_all.csv.gz",index=False)
    before[cols+["balance_selected"]].to_csv(out/"train_before_balance.csv.gz",index=False)
    train[cols].to_csv(out/"train_balanced.csv.gz",index=False)
    val[cols].to_csv(out/"validation.csv.gz",index=False)
    test[cols].to_csv(out/"test_p12.csv.gz",index=False)
    final_manifest = pd.concat([train[cols], val[cols], test[cols]], ignore_index=True)
    final_manifest.to_csv(out/"ldm_dp_manifest.csv.gz", index=False)
    excluded[cols].to_csv(out/"excluded.csv.gz",index=False)
    rows=[]
    for name, part in [("train_before",before),("train_balanced",train),("validation",val),("test",test)]:
        one=part[part.positive_label_count.eq(1)]
        for label in label_cols:
            x=one[one.single_positive_label.eq(label)]
            rows.append({"split":name,"label":label,"images":len(x),"studies":x.study_id.nunique(),"patients":x.subject_id.nunique(),
                         "sufficiency": "sufficient" if x.subject_id.nunique() >= 500 else ("limited" if x.subject_id.nunique() >= 100 else "insufficient")})
    pd.DataFrame(rows).to_csv(out/"class_distribution.csv",index=False)
    summary={"seed":a.seed,"no_finding_ratio":a.no_finding_ratio,"no_finding_cap":cap,
      "image_decode_verified":a.verify_images,"input_sha256":{str(x):digest(x) for x in [sp,cp,rp]},
      "counts":{"all":len(df),"train_before_balance":len(before),"train_balanced":len(train),
      "validation":len(val),"test_p12":len(test),"excluded":len(excluded),"missing_image":int((~df.image_exists).sum()),
      "unreadable_image":int((df.image_exists&~df.image_readable).sum()),"missing_report":int((~df.report_exists).sum()),
      "no_finding_before":len(nf_idx),"no_finding_after":min(len(nf_idx),cap)},
      "patient_overlap":overlap,"test_at_least_10240":len(test)>=10240,
      "split_policy":"p10/p11 official train+validate; p10/p11 official test excluded; p12 all test"}
    (out/"split_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(summary,indent=2,ensure_ascii=False))

if __name__ == "__main__":
    main()
