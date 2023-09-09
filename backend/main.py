import modal


image = modal.Image.debian_slim().pip_install_from_requirements("requirements.txt")

stub = modal.Stub("pair-sports-ml", image=image)


@stub.function()
@modal.web_endpoint(method="GET")
def root():
    return "Hello world"


@stub.function()
@modal.web_endpoint(method="POST")
def video_to_points():
    

if __name__ == "__main__":
    stub.serve()
