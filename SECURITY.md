# 🛡️ Security & Hardening Policy

This document outlines the security controls, hardening practices, and compliance measures applied to this repository prior to deployment and GitHub publishing.

---

### **1. 🔑 Secrets & API Keys**
* **No Hardcoded Secrets**: Scanned and verified that zero API keys, tokens, or credentials are hardcoded in the codebase.
* **Environment Isolation**: All configuration is decoupled via environment variables using `.env.example` as a template.
* **Strict `.gitignore`**: Automatically blocks `.env`, `*.key`, `*.pem`, and `secrets.json` from git tracking.

---

### **2. 🌐 Web & API Security**
* **XSS Defense**: Incoming input text is sanitized using HTML entity escaping (`html.escape`) before reflection or processing.
* **OWASP Security Headers**: Injected into all HTTP responses:
  * `Content-Security-Policy`: Restricts resource execution to trusted local scripts and Google Fonts.
  * `X-Content-Type-Options: nosniff`: Prevents MIME-sniffing exploits.
  * `X-Frame-Options: DENY`: Prevents clickjacking attacks.
  * `X-XSS-Protection: 1; mode=block`: Activates browser XSS filters.
  * `Referrer-Policy: strict-origin-when-cross-origin`: Restricts referrer information.
* **CORS Restrictions**: Access-Control headers are locked down to local origin (`http://127.0.0.1:8501`), blocking unauthorized cross-origin requests.
* **Rate Limiting**: Integrated sliding-window rate limiter restricting traffic to **60 requests/minute per client IP** to prevent Denial of Service (DoS) and abuse.
* **Payload Size Limits**: Max request body clamped at **64 KB** (`HTTP 413 Payload Too Large`) to protect memory buffers from exhaustion attacks.
* **Input Validation**: String type validation and max length truncation at 1,000 characters.

---

### **3. 🚫 Debug Mode & Error Leakage**
* **Debug Mode OFF**: Verbose stack traces and internal exceptions are suppressed.
* **Generic Error Responses**: Failures return standard JSON error payloads (`{"error": "..."}`) without exposing internal file paths or server details.

---

### **4. 📦 Dependencies & File Safety**
* **Clean Dependency Footprint**: Unused libraries removed; minimal production requirements pinned in `requirements.txt`.
* **Large File Exclusions**: Model weights (`model.safetensors` > 1.1 GB) are blocked via `.gitignore` to adhere to GitHub's 100 MB file limit.
* **Cache Cleaned**: `__pycache__`, `.pytest_cache`, and temporary runtime logs excluded from repository.

---

### **5. 🗄️ Database & Authentication**
* **Stateless Architecture**: This inference service operates statelessly in memory. No user passwords, persistent sessions, or personal identifiable information (PII) are stored on disk.
* **Production Best Practice**: If integrated into a database-backed user management system, passwords must be hashed using `bcrypt` or `Argon2id` with proper salting, and admin routes must be guarded by JWT / RBAC middleware.
