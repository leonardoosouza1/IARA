# Base oficial ROCm (Limpa e compatível com PyTorch)
FROM rocm/pytorch:latest

# Metadados
LABEL maintainer="Saci Engineer"
LABEL description="Saci Forex Lab - BitNet Transformer com suporte a RX 6750 XT"

# Variáveis de ambiente
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instala dependências do sistema e TA-Lib
RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Instala TA-Lib da fonte (necessário para o wrapper Python)
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib \
    && ./configure --prefix=/usr \
    && make \
    && make install \
    && cd .. \
    && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Define pasta de trabalho
WORKDIR /app

# Copia requisitos e instala
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Mantém o container vivo
CMD ["tail", "-f", "/dev/null"]
