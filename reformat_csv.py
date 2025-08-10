import csv

# Reformat polynomial predictions file
print("Reformatting polynomial predictions file...")
with open('housing_price_predictions_polynomial.csv', 'r') as infile:
    reader = csv.DictReader(infile)
    data = list(reader)

with open('housing_price_predictions_polynomial_simple.csv', 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['Id', 'SalePrice'])
    for row in data:
        writer.writerow([row['Id'], row['SalePrice']])

print("Created housing_price_predictions_polynomial_simple.csv")

# Reformat linear predictions file
print("Reformatting linear predictions file...")
with open('housing_price_predictions1.csv', 'r') as infile:
    reader = csv.DictReader(infile)
    data = list(reader)

with open('housing_price_predictions_simple.csv', 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['Id', 'SalePrice'])
    for row in data:
        writer.writerow([row['Id'], row['SalePrice']])

print("Created housing_price_predictions_simple.csv")
