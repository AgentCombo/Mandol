Data Format Guide
==================

This document is the authoritative reference for MemoryUnit data formats, explaining the meaning of each field in ``raw_data`` and ``metadata``, how the system processes them, and current limitations.

raw_data Field Reference
-------------------------

``raw_data`` has type ``Dict[str, Any]`` and can store any key-value pairs. However, the system only auto-vectorizes the following fields:

Auto-Vectorized Fields
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 12 30 40

   * - Field Name
     - Type
     - Processing
     - Description
   * - ``text_content``
     - str
     - Auto-generates Dense Embedding + BM25 + Sparse index
     - **Core field**, almost all retrieval and build processes depend on it
   * - ``image_path``
     - str
     - Calls Embedder's ``embed_image_paths`` to generate vectors
     - Interface reserved, current implementation falls back to path string encoding

.. important::

   **Text-first principle**: When both ``text_content`` and ``image_path`` are provided, the system only uses ``text_content`` to generate vectors. A MemoryUnit can only use either text or image, not both.

Text Extraction Fallback Order
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the system needs to extract text from a MemoryUnit, it searches in the following order:

::

   text_content → text → content → summary → title → message → first string value

This means ``raw_data={"content": "some text"}`` will also work, but ``raw_data={"body": "some text"}`` won't be automatically recognized (unless it's the only string value in the dictionary).

.. tip::

   Always use ``text_content`` as the text field name, and avoid relying on the fallback mechanism. The fallback mechanism is designed for internally generated units.

System-Internally Auto-Generated Fields
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

During high-level memory construction, the system automatically creates MemoryUnits with the following ``raw_data`` fields:

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - Field Name
     - Unit Type Where It Appears
     - Description
   * - ``text_content``
     - Entity, Event, Summary
     - Formatted text description
   * - ``entity_name``
     - Entity
     - Entity name
   * - ``entity_type``
     - Entity
     - Entity type (Person / Organization / Location, etc.)
   * - ``description``
     - Entity, Event
     - Detailed description
   * - ``summary``
     - Summary
     - Summary text
   * - ``type``
     - Summary
     - Summary type identifier
   * - ``insights``
     - Insights
     - List of insight contents

User-Defined Fields
~~~~~~~~~~~~~~~~~~~

You can store any custom fields in ``raw_data``. These fields will be stored and serialized, but **not auto-vectorized**:

.. code-block:: python

   MemoryUnit(
       uid=Uid("msg_1"),
       raw_data={
           "text_content": "Zhang San went to Beijing on a business trip",   # Auto-vectorized
           "speaker": "Li Si",                  # Stored only, not vectorized
           "source": "WeChat",                   # Stored only, not vectorized
           "session_id": "s_001",               # Stored only, not vectorized
       },
   )

metadata Field Reference
-------------------------

``metadata`` is for storing additional tags. It doesn't participate in vectorization or text extraction, but can be used for subsequent filtering.

Common metadata key names:

.. list-table::
   :header-rows: 1
   :widths: 22 20 58

   * - Key Name
     - Type
     - Description
   * - ``timestamp``
     - str
     - ISO 8601 format timestamp, auto-filled by the system (if not provided)
   * - ``speaker``
     - str
     - Speaker identifier
   * - ``source``
     - str
     - Data source (WeChat / Email / Document, etc.)
   * - ``session_id``
     - str
     - External session ID

Currently Unsupported Content Formats
--------------------------------------

.. warning::

   The following formats **cannot** be directly inserted into the system as MemoryUnits:

   - **PDF files**: Must be extracted to plain text first, then put into ``text_content``
   - **Markdown files**: Must have formatting markers removed first, then put into ``text_content``
   - **Word / Excel files**: Must be extracted to plain text first
   - **Audio / Video files**: Must be transcribed to text first
   - **Image pixel data**: Currently only image paths (``image_path``) are supported, and the implementation is reserved as an interface

   Processing approach: Use external tools to extract file contents to plain text, then put into ``raw_data["text_content"]``.

.. code-block:: python

   # ❌ Cannot pass file paths directly
   MemoryUnit(uid=Uid("doc_1"), raw_data={"file_path": "/data/report.pdf"})

   # ✅ Extract text first
   text = extract_pdf_text("/data/report.pdf")
   MemoryUnit(uid=Uid("doc_1"), raw_data={"text_content": text})

Complete Example
----------------

.. code-block:: python

   from mandol import MemoryUnit, Uid

   unit = MemoryUnit(
       uid=Uid("msg_001"),
       raw_data={
           "text_content": "Zhang San went to Beijing on a business trip today",
           "image_path": "/photos/trip.jpg",
       },
       metadata={
           "timestamp": "2024-01-15T10:00:00",
           "speaker": "System notification",
           "source": "Calendar management",
       },
   )
