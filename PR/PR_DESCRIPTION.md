# PR: Replace failing Uni-Core pretraining smoke test with unimol_tools-based validation for HPC clusters

**Base**: `deepmodeling/Uni-Mol` ← **Head**: `han-seo-yun/Uni-Mol:ksc-env-setup`
**상태**: KSC GPU 잡 제출 로그 미확인 (재확인 필요 — 아래 "확인 필요" 참고)

## 배경 — 왜 원본 경로를 그대로 쓰지 않았는지

원본 `unimol/` 경로는 Uni-Core(fairseq 기반)로 처음부터 사전학습(pretraining)하는 구조이고, 이건:
- `--enable-cuda-ext`로 Uni-Core를 직접 빌드해야 함 (컴파일 실패 위험이 있는 무거운 단계)
- 사전학습 데이터셋이 필요 (약 114.76GB, 이 프로젝트에는 준비돼 있지 않음)
- 4-GPU 분산학습(`torch.distributed.launch --nproc_per_node=4`)까지 필요

즉, 원본 경로는 "실제 사전학습 데이터가 있고 처음부터 학습하는" 시나리오에만 맞고, 지금 프로젝트가 실제로 필요한 건 기존 사전학습 가중치를 가져와 다운스트림 태스크에 맞게 쓰는 것(fine-tuning)에 더 가깝다. deepmodeling이 별도로 배포하는 `unimol_tools`(PyPI 설치 가능, Uni-Core 의존성 없음, HuggingFace에서 사전학습 가중치 자동 다운로드)가 이 시나리오에 훨씬 적합해서, KSC 첫 GPU 파이프라인 검증은 이 경로로 먼저 진행했다. 원본 Uni-Core 경로는 실제 사전학습 데이터 경로가 준비되면 그대로 되살릴 수 있도록 잡 스크립트 하단에 주석으로 보존해뒀다.

## 변경 내용

- `ksc_job_unimol.pbs` 신규: Slurm(`#SBATCH`) 잡 스크립트. 기본 페이로드는 `unimol_smoketest.py`(unimol_tools 5-fold 분류 스모크 테스트), 원본 Uni-Core `unicore-train` 커맨드는 파일 하단에 주석으로 보존
  - conda는 `module load conda/...`가 아니라 base Miniconda의 `conda.sh`를 직접 source
  - `conda activate` 호출을 `set +u`/`set -u`로 방어적으로 감쌈 (MolE job 890315에서 실제 확인된 activate.d/`set -u` 충돌 문제, 아래 참고)
  - `HF_HOME`을 scratch로 지정
  - `--comment="field=chem;appl=pytorch"`, `--mail-user`/`--mail-type=END,FAIL`
- `unimol_smoketest.py` 신규: 20개 toy SMILES로 `MolTrain(task="classification", ...)` fit까지 end-to-end 확인하는 최소 스크립트

## upstream에 보고할 실제 문제

1. **conda `activate.d` 훅이 `set -u`(nounset)에 안전하지 않음** — MolE 잡(890315)에서 `MKL_INTERFACE_LAYER: unbound variable`로 실제 재현. Uni-Mol 잡에서도 동일 패턴의 환경 구성이면 같은 문제가 날 수 있어 선제적으로 수정.
2. `unimol_tools` README의 quick-start 예시는 `MolTrain.fit(data=<pandas DataFrame>)`가 되는 것처럼 보이지만, `datareader.py`의 `read_data()`는 `str`(csv/sdf 경로)·`dict`·`list`/`ndarray`만 받고 DataFrame은 `ValueError: Unknown data type`으로 거부함 — CSV로 저장 후 경로를 넘겨야 함 (`unimol_smoketest.py`에서 실제로 이렇게 우회).

## 검증 근거

- 로컬 macOS(CPU, Python 3.x)에서 `unimol_smoketest.py` 실행 — 5-fold 중 4-fold 실제 학습 완료 + AUC 계산 성공 (2026-08-22)
- macOS에서는 `unimol_tools.MolTrain`이 내부적으로 `multiprocessing.Pool`을 쓰는데, macOS의 spawn 방식 때문에 `if __name__ == "__main__":` 가드가 필요함을 확인 — Linux/KSC는 fork 방식이라 이 문제가 재현되지 않음(가드는 이미 스크립트에 있어 안전)

## 확인 필요

- `/scratch/r978a07/ksc-models/Uni-Mol`에서 제출한 KSC GPU 잡의 로그(`ksc_job_unimol.pbs.o<jobid>`/`.e<jobid>`)가 아직 확보되지 않음 — 실제로 제출됐는지, 제출됐다면 어떤 job ID였는지 재확인 필요.

## 남은 작업

- [ ] KSC GPU 잡 제출 여부/로그 확인
- [ ] `unimol_smoketest.py` KSC GPU에서 정상 완주 확인
- [ ] 결과 확인 후 이 PR을 실제로 오픈
- [ ] (별도) 교수님께 실제 사전학습(114.76GB, Uni-Core) 경로가 필요한지, fine-tuning 경로로 충분한지 확인 후 필요시 원본 Uni-Core 경로 잡 스크립트도 완성
