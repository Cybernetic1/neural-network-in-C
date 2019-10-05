#!/bin/bash
cat $1 | perl -pe 'BEGIN { binmode STDIN, ":utf8"; } s/(.)/ord($1) < 128 ? $1 : sprintf("\\U%04x", ord($1))/ge;'
