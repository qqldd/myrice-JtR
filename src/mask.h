/*
 * This file is part of John the Ripper password cracker,
 * Copyright (c) 2013 by Myrice
 */

/*
 * Maks mode cracker.
 */

#ifndef _JOHN_MASK_H
#define _JOHN_MASK_H

#include "loader.h"

/*
 * Runs the incremental mode cracker.
 */
extern void do_mask_crack(struct db_main *db, char *param);

#endif
