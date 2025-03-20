#!/bin/bash

# Conda 환경이 활성화되어 있는지 확인합니다.
if [ -z "$CONDA_PREFIX" ]; then
  echo "Conda 환경이 활성화되어 있지 않습니다. 먼저 'conda activate [내 가상환경]'을 실행하세요."
  exit 1
fi

# 설치할 GitHub CLI 버전을 설정합니다.
VERSION="2.30.0"

# GitHub CLI tarball 다운로드 (64-bit Linux용)
wget https://github.com/cli/cli/releases/download/v${VERSION}/gh_${VERSION}_linux_amd64.tar.gz

# 압축 해제
tar -xzf gh_${VERSION}_linux_amd64.tar.gz

# 현재 활성화된 conda 환경의 bin 디렉토리로 gh 바이너리 이동
mv gh_${VERSION}_linux_amd64/bin/gh "$CONDA_PREFIX/bin/"

# 다운로드한 파일 및 압축 폴더 삭제
rm -rf gh_${VERSION}_linux_amd64 gh_${VERSION}_linux_amd64.tar.gz

# 설치 확인
gh --version
