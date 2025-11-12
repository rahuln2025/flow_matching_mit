from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml



def parse_args():
    parser = argparse.ArgumentParser(description='Script to convert image to dist of points')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to YAML config file')
    return parser.parse_args()

def image_to_dist(config_path):

    # Load configuration from YAML file
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Load the image
    image_path = config.get('image').get('filepath')
    image = Image.open(image_path)

    # Get original dimensions
    width, height = image.size
    print(f"Original dimensions: {width}x{height}")

    # Calculate square size (use the larger dimension to avoid losing content)
    square_size = max(width, height)

    # Create a new square image with white background
    square_image = Image.new('RGB', (square_size, square_size), 'black')

    # Calculate position to paste the original image (center it)
    paste_x = (square_size - width) // 2
    paste_y = (square_size - height) // 2

    # Paste the original image onto the square canvas
    square_image.paste(image, (paste_x, paste_y))

    # Convert to grayscale
    gray_image = square_image.convert('L')
    gray_array = np.array(gray_image)

    print(f"Resized dimensions: {gray_array.shape}")

    # Optional: Display the result
    plt.figure(figsize=(8, 8))
    plt.imshow(gray_array, cmap='gray')
    plt.title('1:1 Aspect Ratio')
    plt.axis('off')
    plt.savefig(image_path[:-4] + '_grayscale.png')

    # Create a point distribution based on color
    points = []
    threshold = 10  # Threshold for filtering points

    # Collect points
    height, width = gray_array.shape
    for y in range(height):
        for x in range(width):
            if gray_array[y, x] > threshold:
                points.append((x, y))

    points_array = np.array(points)
    print(points_array.shape)

    # Normalize coordinates to [-1, 1] range
    points_array = points_array.astype(float)
    points_array[:,0] = (points_array[:,0] - width/2) / (width/2)  # x coordinates
    points_array[:,1] = -1*(points_array[:,1] - height/2) / (height/2)  # y coordinates

    # Add noise to create more samples
    num_noise_samples = int(config.get('image').get('num_noise_samples'))
    noise_std = config.get('image').get('noise')  # Reduced noise since we're in normalized space

    # Create noisy copies of the normalized points
    if num_noise_samples > 0:
    
        noisy_points = np.zeros((len(points_array) * num_noise_samples, 2))
        for i in range(num_noise_samples):
            noise = np.random.normal(0, noise_std, points_array.shape)
            noisy_points[i*len(points_array):(i+1)*len(points_array)] = points_array + noise

        # Combine original and noisy points
        all_points = np.vstack([points_array, noisy_points])

    else:
        all_points = points_array

    # Visualize the normalized point distribution
    plt.figure(figsize=(15, 5))

    # Original normalized points
    plt.subplot(121)
    plt.scatter(points_array[:, 0], points_array[:, 1], s=0.5, alpha=0.5, label='Original')
    plt.title('Normalized Point Distribution')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.axis('equal')

    # Points with noise
    plt.subplot(122)
    plt.scatter(all_points[:, 0], all_points[:, 1], s=0.5, alpha=0.2, label='With Noise')
    plt.title('Normalized Point Distribution with Noise')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.axis('equal')

    plt.tight_layout()
    plt.savefig('image_to_points.png')

    # Save the normalized points
    np.savetxt('image_data.txt', all_points, fmt='%f')

    return all_points


if __name__ == "__main__":
    args = parse_args()
    image_to_dist(args.config)
