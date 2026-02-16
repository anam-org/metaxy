??? abstract annotate "Data vs Metadata Clarifications"

    Metaxy features represent tabular **metadata**, typically containing references to external multimodal **data** such as files, images, or videos.

    | **Subject** | **Description** |
    |---------|-------------|
    | **Data** | The actual multimodal data itself, such as images, audio files, video files, text documents, and other raw content that your pipelines process and transform. |
    | **Metadata** | Information about the data, typically including references to where data is stored (e.g., object store keys), plus additional descriptive entries such as video length, file size, format, version, and other attributes. |

    Metaxy does not interact with **data** and is not responsible for its content.
    As an edge case, Metaxy may also manage pure **metadata** tables that do not reference any external **data**.
