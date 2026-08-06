FROM public.ecr.aws/lambda/python:3.12

# Install system dependencies for FAISS
RUN dnf install -y gcc gcc-c++ && dnf clean all

# Install Python dependencies
RUN pip install --no-cache-dir \
    boto3>=1.34.0 \
    faiss-cpu>=1.8.0 \
    numpy>=1.26.0 \
    PyPDF2>=3.0.0

# Copy function code
COPY functions/ ${LAMBDA_TASK_ROOT}/

# Set the handler
CMD ["legal_orchestrator.lambda_handler"]
