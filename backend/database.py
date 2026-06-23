import sqlite3

def criar_base():
    conn = sqlite3.connect("ouviescrevi.db")
    cursor = conn.cursor()

    # Tabela para transcrições
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transcricoes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ficheiro TEXT,
            data TEXT
        )
    """)

    # Tabela para status
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS status (
            id INTEGER PRIMARY KEY,
            manutencao BOOLEAN
        )
    """)

    # Inserir status inicial se não existir
    cursor.execute("INSERT OR IGNORE INTO status (id, manutencao) VALUES (1, 0)")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS visitas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL,
            day TEXT NOT NULL,
            visitor_hash TEXT,
            created_at TEXT NOT NULL
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_visitas_day ON visitas(day)")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS site_content (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()

# Executar ao iniciar
criar_base()
