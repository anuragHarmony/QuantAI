#!/bin/bash
# Download Binance Book Ticker Data for Top 10 Coins

# Top 10 coins by market cap
SYMBOLS=(
    "BTCUSDT"
    "ETHUSDT"
    "BNBUSDT"
    "SOLUSDT"
    "ADAUSDT"
    "XRPUSDT"
    "DOGEUSDT"
    "MATICUSDT"
    "DOTUSDT"
    "AVAXUSDT"
)

# Historical dates (July-August 2024 - more likely to be available)
DATES=(
    "2024-08-01"
    "2024-08-02"
    "2024-08-03"
    "2024-08-04"
    "2024-08-05"
    "2024-07-31"
    "2024-07-30"
    "2024-07-29"
    "2024-07-28"
    "2024-07-27"
)

BASE_URL="https://data.binance.vision/data/futures/um/daily/bookTicker"
OUTPUT_DIR="/home/user/QuantAI/data/binance_book_ticker"

echo "Downloading Binance Book Ticker Data..."
echo "Output directory: $OUTPUT_DIR"
echo ""

# Create directory structure
mkdir -p "$OUTPUT_DIR"

total_files=0
successful_downloads=0

for symbol in "${SYMBOLS[@]}"; do
    echo "Processing $symbol..."
    mkdir -p "$OUTPUT_DIR/$symbol"

    for date in "${DATES[@]}"; do
        filename="${symbol}-bookTicker-${date}.zip"
        url="$BASE_URL/$symbol/$filename"
        output_path="$OUTPUT_DIR/$symbol/$filename"

        # Skip if already downloaded
        if [ -f "$output_path" ]; then
            echo "  ✓ $filename (already exists)"
            ((successful_downloads++))
            ((total_files++))
            continue
        fi

        # Download with curl
        echo -n "  Downloading $filename... "
        if curl -f -s -S -o "$output_path" "$url" 2>/dev/null; then
            echo "✓"
            ((successful_downloads++))
        else
            echo "✗ (not available)"
            rm -f "$output_path"
        fi
        ((total_files++))
    done
    echo ""
done

echo "========================================="
echo "Download Summary:"
echo "  Total files attempted: $total_files"
echo "  Successful downloads: $successful_downloads"
echo "  Output directory: $OUTPUT_DIR"
echo "========================================="

# Unzip all files
echo ""
echo "Extracting downloaded files..."
cd "$OUTPUT_DIR"
find . -name "*.zip" -exec unzip -q -o {} -d $(dirname {}) \;

echo ""
echo "Extraction complete!"
echo ""
echo "Files downloaded:"
find "$OUTPUT_DIR" -name "*.csv" | wc -l
