idf_component_register(
    SRCS
        "main.c"
        "ili9341.c"
        "gfx.c"
        "face.c"
        "uart_cmd.c"
    INCLUDE_DIRS "."
    REQUIRES driver esp_driver_gpio freertos esp_common
)
