# scripts/build_lambda.sh
set -euo pipefail
rm -rf build lambda.zip
mkdir -p build
uv pip install --target build \
  --python-platform x86_64-manylinux2014 --only-binary=:all: \
  lancedb anthropic langgraph pydantic pyarrow
cp -r src/edgar_rag build/
cd build && zip -qr ../lambda.zip . -x '*.pyc' '*__pycache__*' && cd ..
echo "$(du -h lambda.zip | cut -f1) lambda.zip"
