
import logging
from dicom_mcp.dicom_client import DicomClient
from dicom_mcp.config import load_config

# Suppress pynetdicom logging for cleaner output
logging.getLogger('pynetdicom').setLevel(logging.CRITICAL)

try:
    config = load_config('C:/Users/25166122M/dicom/mcp/dicom-mcp/configuration.yaml')
    client = DicomClient(
        host=config.nodes[config.current_node].host,
        port=config.nodes[config.current_node].port,
        calling_aet=config.calling_aet,
        called_aet=config.nodes[config.current_node].ae_title
    )
    # Query for all studies, which is the default for query_study()
    studies = client.query_study()
    print(studies)
except Exception as e:
    print(f"An error occurred: {e}")
