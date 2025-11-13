# QuantAI Refactoring Plan - SOLID Principles & Async

## 🎯 Goals

1. **SOLID Principles**
   - S: Single Responsibility
   - O: Open/Closed
   - L: Liskov Substitution
   - I: Interface Segregation
   - D: Dependency Inversion

2. **Async/Await** - All I/O operations async
3. **URL Fetching** - Support web URLs like Claude
4. **Clean Architecture** - Layers and boundaries
5. **Type Safety** - Full type hints
6. **Testability** - Easy to mock and test

## 📋 Refactoring Checklist

### Phase 1: Core Abstractions ✅ IN PROGRESS
- [ ] Create abstract interfaces
- [ ] Implement dependency injection
- [ ] Add async support to all I/O
- [ ] Separate concerns into layers

### Phase 2: URL Support
- [ ] Add URL fetcher
- [ ] HTML to markdown converter
- [ ] URL validation and sanitization

### Phase 3: Async Everything
- [ ] Async document processing
- [ ] Async embedding generation
- [ ] Async vector store operations
- [ ] Async RAG pipeline

### Phase 4: Testing
- [ ] Unit tests for all components
- [ ] Integration tests
- [ ] Mock implementations

## 🏗️ New Architecture

```
QuantAI/
├── domain/                      # Domain layer (business logic)
│   ├── entities/               # Core entities
│   ├── repositories/           # Repository interfaces
│   └── services/               # Domain services
│
├── application/                # Application layer (use cases)
│   ├── use_cases/             # Application use cases
│   └── interfaces/            # Application interfaces
│
├── infrastructure/             # Infrastructure layer (implementations)
│   ├── repositories/          # Concrete repository implementations
│   ├── services/              # External service implementations
│   └── adapters/              # Adapters for external systems
│
└── presentation/               # Presentation layer (API, CLI)
    ├── api/                   # FastAPI web interface
    └── cli/                   # Command-line interface
```

## 🔧 Key Improvements

### Before (Current)
```python
# Tightly coupled
class KnowledgeEngine:
    def __init__(self):
        self.vector_store = VectorStore()  # Direct instantiation
        self.processor = DocumentProcessor()  # Hard dependency
```

### After (Refactored)
```python
# Dependency injection with interfaces
class KnowledgeEngine:
    def __init__(
        self,
        vector_store: IVectorStore,  # Interface
        processor: IDocumentProcessor  # Interface
    ):
        self._vector_store = vector_store
        self._processor = processor
```

This follows:
- **D**ependency Inversion: Depend on abstractions
- **S**ingle Responsibility: Engine orchestrates, doesn't implement
- **O**pen/Closed: Easy to extend with new implementations

## 📝 Implementation Notes

- Using Python's `abc` module for interfaces
- Type hints everywhere (`typing`, `Protocol`)
- Async I/O with `asyncio` and `aiohttp`
- Dependency injection container (optional: `dependency-injector`)

Status: **IN PROGRESS** - Starting refactoring now
