#!/bin/bash

show_help() {
    echo "How to use tools:"
    echo ""
    echo "Options:"
    echo "  -m, --mode <plot|load>      Choice a mode [plot|load]"
    echo "  -c --cache <true|false>     Choice a cache [true|false] - default is true" 
    echo "  -i, --images <number>       The number of images will be show - default is 10"
    echo "  -h, --help                  Show help"
    echo ""
    echo "For instance:"
    echo "  $0 -m plot -i 10"
    echo "  $0 -m load -c false"
    exit 1
}

MODE="plot"
IMAGES=10
IS_CACHE=true

if [ $# -eq 0 ]; then
    show_help
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--mode)
            MODE="$2"
            if [[ "$MODE" != "plot" && "$MODE" != "load" ]]; then
                echo "Mode must be 'plot' or 'load'"
                exit 1
            fi
            shift 2
            ;;
        -i|--images)
            IMAGES="$2"
            if ! [[ "$IMAGES" =~ ^[0-9]+$ ]]; then
                echo "Images must be a number"
                exit 1
            fi
            shift 2
            ;;
        -c|--cache)
            IS_CACHE="$2"
            if [[ "$IS_CACHE" != "true" && "$IS_CACHE" != "false" ]]; then
                echo "Cache must be 'true' or 'false'"
                exit 1
            fi
            shift 2
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Not found option '$1'"
            show_help
            ;;
    esac
done

CMD="python main.py --mode $MODE --images $IMAGES --cache $IS_CACHE"
eval $CMD