import random

# Generate 100 unique random numbers between 1 and 100
random_numbers = random.sample(range(1, 101), 100)

# Save to a text file
with open("data.txt", "w") as file:
    for num in random_numbers:
        file.write(f"{num}\n")

print("Random numbers saved to random_numbers.txt!")