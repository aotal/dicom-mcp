
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
    instances = client.query_instance(series_instance_uid='1.3.46.670589.30.41.0.1.128635482625724.1743412778135.1')
    print(instances)
except Exception as e:
    print(f"An error occurred: {e}")
