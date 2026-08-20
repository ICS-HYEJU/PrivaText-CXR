# MIMIC-CXR-JPG 데이터 처리 메커니즘

이 문서는 Claude가 p10/p11 학습 및 p12 custom test holdout을 재현할 때 사용하는
단일 가이드다. 현재 DICOM 파이프라인은 삭제하거나 대체하지 않았다. JPG 경로는
`--manifest_csv`를 전달한 경우에만 활성화된다.

## 데이터 위치

- report/metadata: `/storage/hjchoi/physionet.org/files/mimic-cxr/2.1.0`
- JPG: `/storage/hjchoi/physionet.org/files/mimic-cxr-jpg/2.1.0`
- JPG 실제 구조: `<jpg_root>/pXX/pXXXXXXXX/sYYYYYYYY/<dicom_id>.jpg`
- report 구조: `<report_root>/files/pXX/pXXXXXXXX/sYYYYYYYY.txt`

`IMAGE_FILENAMES`와 record-list의 경로에는 `files/`가 있지만 JPG 루트의 실제
디렉터리에는 해당 단계가 없다. 경로를 그대로 join하지 말고 ID 컬럼으로 조립한다.

## Pairing

- image/split/record-list 결합 키: `(dicom_id, subject_id, study_id)`
- report 및 CheXpert 결합 키: `(subject_id, study_id)`
- 하나의 study에 여러 view가 있으면 report와 CheXpert label을 공유한다.
- 환자 누출 검사는 반드시 `subject_id` 집합으로 수행한다.

## Custom split

- train: p10/p11 중 공식 `train`
- validation: p10/p11 중 공식 `validate`
- 제외: p10/p11 중 공식 `test`
- test: p12 전체(공식 split 값과 무관)
- 결과 보고 시 공식 MIMIC test가 아니라 **prefix-based custom holdout**이라고 명시한다.

현재 검증된 usable 수는 train 72,249, validation 802, p12 test 37,190이다.
환자 교집합은 모두 0이며 p12 test는 10,240장 조건을 충족한다.

## No Finding 조정

validation과 test는 원 분포를 유지한다. train의 single-positive `No Finding`만
seed 42로 선택한다. 기본 상한은 가장 큰 abnormal single-positive 클래스의 2배다.
현재 Lung Opacity 3,115장의 2배인 6,230장으로 제한되어 balanced train은
53,857장이다. 원본 후보는 `train_before_balance.csv.gz`에 보존된다.

## Manifest 재생성

```bash
docker exec CXR_dp_medclip sh -lc '
  cd /workspace/PrivaText-CXR &&
  /opt/conda/bin/python -B src/Data/prepare_mimic_jpg.py \
    --report_root /storage/hjchoi/physionet.org/files/mimic-cxr/2.1.0 \
    --jpg_root /storage/hjchoi/physionet.org/files/mimic-cxr-jpg/2.1.0 \
    --out_dir data_manifests/mimic_p10_p12 \
    --seed 42 --no_finding_ratio 2.0 --verify_images'
```

최종 학습 입력은 `data_manifests/mimic_p10_p12/ldm_dp_manifest.csv.gz`다.
원본 CSV checksum과 검증 결과는 `split_summary.json`, 단일-label 분포는
`class_distribution.csv`에 기록된다.

## Fine-tuning

기존 DICOM 실행은 `--manifest_csv`를 생략하면 이전과 동일하다. JPG 학습만 다음
옵션을 추가한다.

```bash
docker exec CXR_dp_medclip sh -lc '
  cd /workspace/PrivaText-CXR &&
  /opt/conda/bin/python src/LDM_dp_finetune.py \
    --root_path /storage/hjchoi/physionet.org/files/mimic-cxr/2.1.0 \
    --manifest_csv data_manifests/mimic_p10_p12/ldm_dp_manifest.csv.gz \
    --split train \
    --text_mode LABEL+IMPRESSION \
    <나머지 학습 옵션>'
```

`LABEL+IMPRESSION`은 positive CheXpert label이 없는 행을 로더에서 추가 제외한다.
현재 수는 train 49,907, validation 748, test 35,107이다.

## 완료 검증

- 모든 존재 JPG decode 성공
- manifest의 `dicom_id` 중복 0
- train/validation/test 환자 교집합 0
- p10/p11 공식 test가 train에 0
- test prefix는 p12만 존재
- p12 usable test 10,240 이상
- JPG tensor는 `float32 [1,256,256]`, 범위 `[-1,1]`
- `./init.sh` 통과

다운로드가 추가되면 manifest를 재생성한다. 현재 누락 JPG 1,403개 중 p10이
1,374개, p11이 22개, p12가 7개이므로 다운로드 상태가 바뀌면 개수와 No Finding
상한도 달라질 수 있다.
