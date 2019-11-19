#!/usr/bin/python3
import buildModel

if __name__ == "__main__":
    shape = (1000, 28, 28, 1)
    generate(1, shape)
def generate(images, shape):
    model = buildModel.create_model()
    generate_images = images
    return generate_images
