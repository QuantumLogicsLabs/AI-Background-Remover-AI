from low_light_enhancement import enhance_low_light_from_path
import json

if __name__ == "__main__":
    result = enhance_low_light_from_path(
        "test_images/dog_dark.jpg",
        output_path="test_images/dog_dark_enhanced.jpg"
    )
    print(json.dumps(result, indent=2))