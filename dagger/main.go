package main

import (
	"context"
	"log"
)

func main() {
	ctx := context.Background()

	if err := Pipeline(ctx); err != nil {
		log.Fatal(err)
	}
}
