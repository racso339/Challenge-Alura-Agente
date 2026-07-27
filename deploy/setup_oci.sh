#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup_oci.sh — Preparar la VM de OCI Compute (Ubuntu 22.04) para el agente.
# Ejecutar DENTRO de la VM, después de clonar el repo y hacer `cd` a la carpeta.
# ---------------------------------------------------------------------------
set -e

echo "==> 1/4 Instalando dependencias del sistema..."
sudo apt update
sudo apt install -y python3-venv python3-pip git netfilter-persistent

echo "==> 2/4 Abriendo el puerto 8501 en el firewall del SO (iptables)..."
# OCI trae iptables MUY restrictivo por defecto: hay que abrir el puerto aquí,
# además de en la Security List de la consola de OCI.
sudo iptables -I INPUT 6 -p tcp --dport 8501 -j ACCEPT
sudo netfilter-persistent save

echo "==> 3/4 Creando entorno virtual e instalando requirements..."
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "==> 4/4 Listo. Pasos que faltan (manuales):"
echo "   a) Guardá tu API key:   echo 'GEMINI_API_KEY=tu_key' > .env"
echo "   b) Generá el índice:    set -a && source .env && set +a && python ingest.py"
echo "   c) Probá el arranque:   streamlit run app.py --server.address 0.0.0.0 --server.port 8501"
echo "   d) Para dejarlo 24/7:   instalá el servicio systemd (ver deploy/rag-pegasus.service y README)."
