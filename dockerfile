# Dockerfile
FROM public.ecr.aws/lambda/python:3.12

COPY requirements-lambda.txt .
RUN pip install --no-cache-dir -r requirements-lambda.txt

COPY src/edgar_rag ${LAMBDA_TASK_ROOT}/edgar_rag
RUN chmod -R a+rX ${LAMBDA_TASK_ROOT}/edgar_rag
CMD ["edgar_rag.lambda_handler.handler"]