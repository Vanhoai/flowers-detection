#!/bin/bash

show_help() {
    echo "How to use:"
    echo ""
    echo "Options:"
    echo "  -f, --framework <keras|pytorch>  Choice a framework [keras|pytorch] - default is pytorch"
    echo "  -m, --mode <train|finetune>      Choice a mode [train|finetune] - default is train"
    echo "  -l, --layers <number>            The number of layers to unfreeze if mode is finetune - default is 5"
    echo "  -h, --help                       Show help"
    echo ""
    echo "For instance:"
    echo "  $0 -f pytorch -m train"
    echo "  $0 -f keras -m finetune -l 5"
    exit 1
}

FRAMEWORK="pytorch"
MODE="train"
LAYERS=5

if [ $# -eq 0 ]; then
    show_help
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        -f|--framework)
            FRAMEWORK="$2"
            if [[ "$FRAMEWORK" != "keras" && "$FRAMEWORK" != "pytorch" ]]; then
                echo "Framework must be 'keras' or 'pytorch'"
                exit 1
            fi
            shift 2
            ;;
        -m|--mode)
            MODE="$2"
            if [[ "$MODE" != "train" && "$MODE" != "finetune" ]]; then
                echo "Mode must be 'train' or 'finetune'"
                exit 1
            fi
            shift 2
            ;;
        -l|--layers)
            LAYERS="$2"
            if ! [[ "$LAYERS" =~ ^[0-9]+$ ]]; then
                echo "Layers must be an integer"
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

echo "Framework: $FRAMEWORK"
echo "Mode: $MODE"
if [ "$MODE" == "finetune" ]; then
    echo "Layers will be unfrozen: $LAYERS"
fi

CMD="python main.py --framework $FRAMEWORK"
if [ "$MODE" == "finetune" ]; then
    CMD="$CMD --mode finetune --layers $LAYERS"
else
    CMD="$CMD --mode train"
fi

eval $CMD