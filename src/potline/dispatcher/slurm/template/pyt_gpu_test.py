"""
PyTorch GPU test.
"""

if __name__ == '__main__':
    import torch
    import time
    import logging
    import sys

    # Configure logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    def wait_for_gpu(timeout=3600) -> bool:
        """
        Wait for GPU to be available.
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if torch.cuda.is_available():
                logger.info("GPU is available.")
                return True
            logger.info("Waiting for GPU to be available...")
            time.sleep(5)
        logger.warning("Timeout reached. No GPU available.")
        return False

    if wait_for_gpu():
        logger.info("Num GPUs Available: %d", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            logger.info("GPU %d: %s", i, torch.cuda.get_device_name(i))
    logger.info("CUDA available: %s", torch.cuda.is_available())
