// Browser-native memory store.
// IndexedDB replaces Python/SQLite for the website version.

const DB_NAME = "PrototypeMemory";
const DB_VERSION = 1;
const STORE = "memories";

export class PrototypeMemory {
  constructor() {
    this.dbPromise = new Promise((resolve, reject) => {
      const request = indexedDB.open(DB_NAME, DB_VERSION);
      request.onupgradeneeded = () => {
        const db = request.result;
        if (!db.objectStoreNames.contains(STORE)) {
          const store = db.createObjectStore(STORE, { keyPath: "id", autoIncrement: true });
          store.createIndex("type", "memory_type");
          store.createIndex("created", "created_at");
        }
      };
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  async remember(content, memory_type = "experience", metadata = {}) {
    const db = await this.dbPromise;
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE, "readwrite");
      const req = tx.objectStore(STORE).add({
        content: String(content),
        memory_type,
        metadata,
        created_at: new Date().toISOString(),
        importance: 0.5,
        confidence: 0.5,
      });
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  }

  async all(limit = 100) {
    const db = await this.dbPromise;
    return new Promise((resolve, reject) => {
      const req = db.transaction(STORE).objectStore(STORE).getAll();
      req.onsuccess = () => resolve(req.result.slice(-limit));
      req.onerror = () => reject(req.error);
    });
  }
}
