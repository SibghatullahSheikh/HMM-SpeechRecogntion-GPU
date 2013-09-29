#!/bin/bash
svn delete $( svn status | sed -e '/^!/!d' -e 's/^!//' )
svn add $( svn status | sed -e '/^?/!d' -e 's/^?//' )
