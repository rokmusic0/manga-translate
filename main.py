from paddleocr import PaddleOCRVL

pipeline = PaddleOCRVL()
output = pipeline.predict("./images/yatsuba.png")
for res in output:
    res.print()
    res.save_to_json(save_path="output")
    res.save_to_markdown(save_path="output")
