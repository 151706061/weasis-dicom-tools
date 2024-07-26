/*
 * Copyright (c) 2024 Weasis Team and other contributors.
 *
 * This program and the accompanying materials are made available under the terms of the Eclipse
 * Public License 2.0 which is available at https://www.eclipse.org/legal/epl-2.0, or the Apache
 * License, Version 2.0 which is available at https://www.apache.org/licenses/LICENSE-2.0.
 *
 * SPDX-License-Identifier: EPL-2.0 OR Apache-2.0
 */
package org.weasis.dicom.ref;

import static org.weasis.dicom.ref.CodingScheme.DCM;
import static org.weasis.dicom.ref.CodingScheme.FMA;
import static org.weasis.dicom.ref.CodingScheme.SCT;

public enum SurfacePart implements AnatomicItem {
  ANTERIOR_TRIANGLE_OF_NECK(SCT, 182329002, 41, 0, 42),
  CORNEA(SCT, 28726007, 109, 0, 108),
  EYELASH(SCT, 85803001, 105, 0, 104),
  FEMALE_EXTERNAL_URETHRAL_ORIFICE(SCT, 279479008, 0, 504, 0),
  FRENULUM_OF_LABIA_MINORA(SCT, 279867004, 0, 508, 0),
  GROIN_SKIN_CREASE(SCT, 280387007, 519, 0, 518),
  HAIR(SCT, 386045008, 0, 0, 0),
  IRIS(SCT, 41296002, 109, 0, 108),
  MALE_EXTERNAL_URETHRAL_ORIFICE(SCT, 279478000, 0, 513, 0),
  MUCOSA_OF_DORSUM_OF_ORAL_PART_OF_TONGUE(FMA, 281534, 157, 0, 154),
  MUCOSA_OF_DORSUM_OF_PHARYNGEAL_PART_OF_TONGUE(FMA, 281537, 155, 0, 152),
  MUCOSA_OF_FLOOR_OF_MOUTH(SCT, 36152006, 161, 0, 158),
  MUCOSA_OF_LOWER_LIP(SCT, 46353006, 165, 0, 162),
  MUCOSA_OF_MANDIBULAR_GINGIVA(SCT, 245823002, 163, 0, 160),
  MUCOSA_OF_MAXILLARY_GINGIVA(SCT, 245814000, 145, 0, 144),
  MUCOSA_OF_ORAL_SEGMENT_OF_HARD_PALATE(FMA, 289677, 147, 0, 146),
  MUCOSA_OF_PALATOGLOSSAL_ARCH(FMA, 60031, 151, 0, 148),
  MUCOSA_OF_PHARYNX(FMA, 55031, 0, 0, 0),
  MUCOSA_OF_POSTERIOR_WALL_OF_OROPHARYNX(FMA, 55060, 153, 0, 150),
  MUCOSA_OF_TIP_OF_TONGUE(SCT, 245831007, 159, 0, 156),
  MUCOSA_OF_TONGUE(SCT, 8001006, 0, 0, 0),
  MUCOSA_OF_UPPER_LIP(SCT, 18444004, 143, 0, 142),
  MUCOSA_OF_UVULA(FMA, 60030, 149, 0, 149),
  NAIL_UNIT_OF_FIFTH_TOE(SCT, 770820003, 441, 0, 440),
  NAIL_UNIT_OF_FINGER(SCT, 770809003, 0, 0, 0),
  NAIL_UNIT_OF_FOURTH_TOE(SCT, 770821004, 439, 0, 438),
  NAIL_UNIT_OF_GREAT_TOE(SCT, 770822006, 433, 0, 432),
  NAIL_UNIT_OF_INDEX_FINGER(SCT, 770815003, 329, 0, 328),
  NAIL_UNIT_OF_LITTLE_FINGER(SCT, 770818001, 335, 0, 334),
  NAIL_UNIT_OF_MIDDLE_FINGER(SCT, 770816002, 331, 0, 330),
  NAIL_UNIT_OF_RING_FINGER(SCT, 770817006, 333, 0, 332),
  NAIL_UNIT_OF_SECOND_TOE(SCT, 770823001, 435, 0, 434),
  NAIL_UNIT_OF_THIRD_TOE(SCT, 770825008, 437, 0, 436),
  NAIL_UNIT_OF_THUMB(SCT, 770810008, 327, 0, 326),
  NAIL_UNIT_OF_TOE(SCT, 770805009, 0, 0, 0),
  ORAL_MUCOSA(SCT, 113277000, 0, 0, 0),
  POSTERIOR_COMMISSURE_OF_LABIUM_MAJORUM(SCT, 4019005, 0, 0, 0),
  RETINA(SCT, 5665001, 0, 0, 0),
  SCLERA(SCT, 18619003, 111, 0, 110),
  SKIN(SCT, 39937001, 0, 0, 0),
  SKIN_OF_ABDOMEN(SCT, 75093004, 0, 0, 0),
  SKIN_OF_ALA_NASI(SCT, 68598004, 23, 0, 24),
  SKIN_OF_ANTECUBITAL_FOSSA(SCT, 17957002, 303, 0, 302),
  SKIN_OF_ANTERIOR_HELIX_OF_EAR(DCM, 130305, 119, 0, 118),
  SKIN_OF_ANTERIOR_PORTION_OF_NECK(SCT, 11584001, 0, 60, 0),
  SKIN_OF_ANTERIOR_SURFACE_OF_FOREARM(SCT, 70559009, 305, 0, 304),
  SKIN_OF_ANTERIOR_SURFACE_OF_KNEE(SCT, 181553006, 405, 0, 404),
  SKIN_OF_ANTERIOR_SURFACE_OF_LOWER_LEG(SCT, 25763004, 407, 0, 406),
  SKIN_OF_ANTERIOR_SURFACE_OF_THIGH(SCT, 61248009, 403, 0, 402),
  SKIN_OF_ANTERIOR_SURFACE_OF_THORAX(SCT, 244106003, 0, 0, 0),
  SKIN_OF_ANTERIOR_SURFACE_OF_UPPER_ARM(SCT, 45981001, 301, 0, 300),
  SKIN_OF_ANTERIOR_TRUNK(SCT, 181491009, 0, 0, 0),
  SKIN_OF_ANTITRAGUS(SCT, 38407007, 123, 0, 122),
  SKIN_OF_ANUS(SCT, 59112000, 0, 512, 0),
  SKIN_OF_AREOLA(SCT, 72005009, 207, 0, 206),
  SKIN_OF_AXILLA(SCT, 76261009, 355, 0, 354),
  SKIN_OF_BACK(SCT, 66643007, 0, 0, 0),
  SKIN_OF_BACK_OF_TRUNK(FMA, 49943, 0, 0, 0),
  SKIN_OF_BACK_OF_UPPER_THORACIC_REGION(SCT, 699893008, 225, 0, 224),
  SKIN_OF_BUTTOCK(SCT, 22180002, 231, 0, 230),
  SKIN_OF_CARUNCLE_OF_EYE(DCM, 130306, 0, 0, 0),
  SKIN_OF_CAVITY_OF_CONCHA(SCT, 51098001, 125, 0, 124),
  SKIN_OF_CHEEK(SCT, 36141000, 13, 0, 14),
  SKIN_OF_CHIN(SCT, 23747009, 35, 58, 36),
  SKIN_OF_CLITORIS(SCT, 29353003, 0, 502, 0),
  SKIN_OF_CRUS_OF_HELIX(SCT, 57726007, 117, 0, 116),
  SKIN_OF_DIGIT_OF_HAND(SCT, 244169007, 0, 0, 0),
  SKIN_OF_DORSAL_AREA_OF_WRIST(SCT, 52876008, 313, 0, 312),
  SKIN_OF_DORSAL_PART_OF_FIFTH_TOE(FMA, 37885, 431, 0, 430),
  SKIN_OF_DORSAL_PART_OF_FOURTH_TOE(FMA, 37882, 429, 0, 428),
  SKIN_OF_DORSAL_PART_OF_GREAT_TOE(FMA, 37873, 423, 0, 422),
  SKIN_OF_DORSAL_PART_OF_INDEX_FINGER(FMA, 38324, 325, 0, 324),
  SKIN_OF_DORSAL_PART_OF_LITTLE_FINGER(FMA, 38333, 319, 0, 318),
  SKIN_OF_DORSAL_PART_OF_MIDDLE_FINGER(FMA, 38327, 323, 0, 322),
  SKIN_OF_DORSAL_PART_OF_RING_FINGER(FMA, 38330, 321, 0, 320),
  SKIN_OF_DORSAL_PART_OF_SECOND_TOE(FMA, 37876, 425, 0, 424),
  SKIN_OF_DORSAL_PART_OF_THIRD_TOE(FMA, 37879, 427, 0, 426),
  SKIN_OF_DORSAL_PART_OF_THUMB(FMA, 38321, 317, 0, 316),
  SKIN_OF_DORSUM_OF_NOSE(FMA, 59532, 19, 53, 20),
  SKIN_OF_EAR(SCT, 1902009, 0, 0, 0),
  SKIN_OF_EAR_LOBULE(SCT, 2059009, 131, 0, 130),
  SKIN_OF_EPIGASTRIC_AREA(SCT, 30598005, 233, 0, 233),
  SKIN_OF_EXTERNAL_AUDITORY_CANAL(SCT, 86409001, 0, 0, 0),
  SKIN_OF_EXTERNAL_GENITALIA(SCT, 60944009, 0, 0, 0),
  SKIN_OF_EYE_REGION(SCT, 362916000, 0, 0, 0),
  SKIN_OF_EYEBROW(SCT, 367577003, 101, 0, 100),
  SKIN_OF_FACE(SCT, 73897004, 0, 0, 0),
  SKIN_OF_FOOT(SCT, 60496002, 0, 0, 0),
  SKIN_OF_FOREHEAD(SCT, 68698007, 7, 52, 8),
  SKIN_OF_GLANS_PENIS(SCT, 7991003, 511, 0, 511),
  SKIN_OF_GLUTEAL_FOLD(SCT, 63029009, 0, 238, 0),
  SKIN_OF_HAND(SCT, 33712006, 0, 0, 0),
  SKIN_OF_HEAD(SCT, 70762009, 0, 0, 0),
  SKIN_OF_HEEL(SCT, 84607009, 463, 0, 460),
  SKIN_OF_HELIX_OF_EAR(SCT, 79313003, 0, 0, 0),
  SKIN_OF_HYPOGASTRIC_REGION(SCT, 367578008, 235, 0, 235),
  SKIN_OF_HYPOTHENAR_REGION_OF_PALM(SCT, 89784008, 343, 0, 342),
  SKIN_OF_INFERIOR_HELIX_OF_EAR(DCM, 130307, 119, 0, 118),
  SKIN_OF_INFERIOR_POSTERIOR_SURFACE_OF_THE_PINNA(DCM, 130308, 139, 0, 138),
  SKIN_OF_INFRAALAR_GROOVE(DCM, 130312, 25, 0, 26),
  SKIN_OF_INFRACLAVICULAR_REGION(SCT, 66288003, 203, 0, 202),
  SKIN_OF_INGUINAL_REGION(SCT, 39687006, 223, 0, 222),
  SKIN_OF_INTERTRAGAL_INCISURE(SCT, 45591000, 129, 0, 128),
  SKIN_OF_JAWLINE(SCT, 244097004, 37, 0, 38),
  SKIN_OF_LABIUM(SCT, 73058008, 0, 0, 0),
  SKIN_OF_LABIUM_MAJUS(SCT, 128252004, 0, 0, 0),
  SKIN_OF_LABIUM_MINUS(SCT, 128253009, 515, 0, 514),
  SKIN_OF_LATERAL_ASPECT_OF_ANKLE(SCT, 181564009, 415, 0, 414),
  SKIN_OF_LATERAL_BORDER_OF_SOLE_OF_FOOT(SCT, 35739000, 461, 0, 462),
  SKIN_OF_LATERAL_CANTHUS(SCT, 37671003, 169, 0, 166),
  SKIN_OF_LATERAL_PART_OF_DORSUM_OF_FOOT(DCM, 130309, 419, 0, 418),
  SKIN_OF_LATERAL_PART_OF_HEEL(SCT, 699909001, 417, 0, 416),
  SKIN_OF_LATERAL_PORTION_OF_NECK(SCT, 5272005, 43, 0, 44),
  SKIN_OF_LIP(SCT, 88089004, 0, 0, 0),
  SKIN_OF_LOWER_ABDOMEN(SCT, 699914002, 221, 0, 220),
  SKIN_OF_LOWER_ANTIHELIX_OF_EAR(DCM, 130310, 0, 0, 0),
  SKIN_OF_LOWER_BACK(SCT, 113182001, 229, 0, 228),
  SKIN_OF_LOWER_CHEST_WALL(SCT, 699915001, 217, 0, 216),
  SKIN_OF_LOWER_EXTREMITY(SCT, 371304004, 0, 0, 0),
  SKIN_OF_LOWER_EYELID(SCT, 40069000, 115, 0, 114),
  SKIN_OF_LOWER_EYELID_MARGIN(DCM, 130311, 113, 0, 112),
  SKIN_OF_LOWER_INNER_QUADRANT_OF_BREAST(FMA, 61427, 213, 0, 212),
  SKIN_OF_LOWER_LIP(SCT, 66934001, 0, 0, 0),
  SKIN_OF_LOWER_OUTER_QUADRANT_OF_BREAST(FMA, 61423, 215, 0, 214),
  SKIN_OF_LOWER_PARASPINAL_REGION(DCM, 130304, 0, 236, 0),
  SKIN_OF_MEDIAL_ASPECT_OF_ANKLE(SCT, 181563003, 443, 0, 442),
  SKIN_OF_MEDIAL_BORDER_OF_SOLE_OF_FOOT(SCT, 52953006, 459, 0, 458),
  SKIN_OF_MEDIAL_CANTHUS(SCT, 27887005, 167, 0, 164),
  SKIN_OF_MEDIAL_PART_OF_DORSUM_OF_FOOT(DCM, 130313, 421, 0, 420),
  SKIN_OF_MEDIAL_PART_OF_HEEL(SCT, 699919007, 445, 0, 444),
  SKIN_OF_MEDIAL_SURFACE_OF_THIGH(SCT, 73958006, 401, 0, 400),
  SKIN_OF_MID_BACK(DCM, 130323, 227, 0, 226),
  SKIN_OF_MID_PARASPINAL_REGION(DCM, 130303, 0, 234, 0),
  SKIN_OF_NASOLABIAL_FOLD(SCT, 37108007, 27, 0, 28),
  SKIN_OF_NECK(SCT, 43081002, 0, 0, 0),
  SKIN_OF_NIPPLE(SCT, 54468004, 205, 0, 204),
  SKIN_OF_NOSE(SCT, 113179006, 0, 0, 0),
  SKIN_OF_NUCHAL_REGION(SCT, 4658004, 45, 0, 46),
  SKIN_OF_OCCIPITAL_REGION(SCT, 79951008, 1, 61, 2),
  SKIN_OF_PALM_OF_HAND(SCT, 70887009, 341, 0, 340),
  SKIN_OF_PALMAR_AREA_OF_WRIST(SCT, 24527008, 337, 0, 336),
  SKIN_OF_PALMAR_PART_OF_INDEX_FINGER(FMA, 38344, 347, 0, 346),
  SKIN_OF_PALMAR_PART_OF_LITTLE_FINGER(FMA, 38357, 353, 0, 352),
  SKIN_OF_PALMAR_PART_OF_MIDDLE_FINGER(FMA, 38347, 349, 0, 348),
  SKIN_OF_PALMAR_PART_OF_RING_FINGER(FMA, 38354, 351, 0, 350),
  SKIN_OF_PALMAR_PART_OF_THUMB(FMA, 38341, 345, 0, 344),
  SKIN_OF_PARANASAL_CHEEK(DCM, 130314, 15, 0, 16),
  SKIN_OF_PARASPINAL_AREA_OF_THE_NECK(DCM, 130300, 0, 62, 0),
  SKIN_OF_PARASPINAL_AREA_OF_THE_SUPERIOR_BACK(DCM, 130301, 0, 63, 0),
  SKIN_OF_PARIETAL_REGION(SCT, 21672008, 3, 0, 4),
  SKIN_OF_PART_OF_DORSAL_SURFACE_OF_HAND(SCT, 281642007, 315, 0, 314),
  SKIN_OF_PENIS(SCT, 35900000, 0, 0, 0),
  SKIN_OF_PERINEUM(SCT, 48014002, 0, 510, 0),
  SKIN_OF_PERIORAL_REGION_OF_FACE(SCT, 110488009, 0, 0, 0),
  SKIN_OF_PHILTRUM(SCT, 84365009, 0, 55, 0),
  SKIN_OF_PLANTAR_PART_OF_FIFTH_TOE(FMA, 38119, 455, 0, 454),
  SKIN_OF_PLANTAR_PART_OF_FOURTH_TOE(FMA, 38116, 453, 0, 452),
  SKIN_OF_PLANTAR_PART_OF_GREAT_TOE(FMA, 38107, 447, 0, 446),
  SKIN_OF_PLANTAR_PART_OF_SECOND_TOE(FMA, 38110, 449, 0, 448),
  SKIN_OF_PLANTAR_PART_OF_THIRD_TOE(FMA, 38113, 451, 0, 450),
  SKIN_OF_POPLITEAL_FOSSA(SCT, 84507004, 411, 0, 410),
  SKIN_OF_POSTAURICULAR_REGION(SCT, 24483006, 9, 0, 10),
  SKIN_OF_POSTERIOR_HELIX_OF_EAR(DCM, 130315, 135, 0, 134),
  SKIN_OF_POSTERIOR_LOBULE_OF_THE_EAR(DCM, 130316, 141, 0, 140),
  SKIN_OF_POSTERIOR_SURFACE_OF_ELBOW(SCT, 181536004, 309, 0, 308),
  SKIN_OF_POSTERIOR_SURFACE_OF_FOREARM(SCT, 41550009, 311, 0, 310),
  SKIN_OF_POSTERIOR_SURFACE_OF_LOWER_LEG(SCT, 47224004, 413, 0, 412),
  SKIN_OF_POSTERIOR_SURFACE_OF_THIGH(SCT, 4578000, 409, 0, 408),
  SKIN_OF_POSTERIOR_SURFACE_OF_THORAX(SCT, 244111001, 49, 0, 50),
  SKIN_OF_POSTERIOR_SURFACE_OF_UPPER_ARM(SCT, 72939005, 307, 0, 306),
  SKIN_OF_PREAURICULAR_REGION(SCT, 86719006, 11, 0, 12),
  SKIN_OF_PREPUCE_OF_CLITORIS(SCT, 76723005, 0, 500, 0),
  SKIN_OF_ROOT_OF_PENIS(SCT, 244117002, 0, 501, 0),
  SKIN_OF_SCALP(SCT, 43067004, 0, 0, 0),
  SKIN_OF_SCROTUM(SCT, 81992007, 505, 0, 503),
  SKIN_OF_SHAFT_OF_PENIS(SCT, 244118007, 0, 507, 0),
  SKIN_OF_SIDE_OF_NOSE(SCT, 314395006, 17, 0, 18),
  SKIN_OF_SOLE_OF_FOREFOOT(DCM, 130317, 457, 0, 456),
  SKIN_OF_SUBMENTAL_AREA(SCT, 34926004, 0, 59, 0),
  SKIN_OF_SUPERIOR_ANTIHELIX_OF_EAR(DCM, 130318, 121, 0, 120),
  SKIN_OF_SUPERIOR_POSTERIOR_HELIX_OF_EAR(DCM, 130319, 133, 0, 132),
  SKIN_OF_SUPERIOR_POSTERIOR_SURFACE_OF_THE_PINNA(DCM, 130320, 137, 0, 136),
  SKIN_OF_SUPRACLAVICULAR_REGION_OF_NECK(SCT, 76072005, 47, 0, 48),
  SKIN_OF_TEMPORAL_REGION(SCT, 16621002, 5, 0, 6),
  SKIN_OF_THENAR_REGION_OF_PALM(SCT, 26795005, 339, 0, 338),
  SKIN_OF_TIP_OF_NOSE(SCT, 79283007, 21, 54, 22),
  SKIN_OF_TOE(SCT, 52034004, 0, 0, 0),
  SKIN_OF_TRAGUS(SCT, 79502000, 127, 0, 126),
  SKIN_OF_UMBILICUS(SCT, 315003, 0, 200, 0),
  SKIN_OF_UPPER_ABDOMEN(SCT, 699935000, 219, 0, 218),
  SKIN_OF_UPPER_ANTIHELIX_OF_EAR(DCM, 130321, 0, 0, 0),
  SKIN_OF_UPPER_EXTREMITY(SCT, 371311000, 0, 0, 0),
  SKIN_OF_UPPER_EYELID(SCT, 41310005, 0, 0, 104),
  SKIN_OF_UPPER_EYELID_MARGIN(DCM, 130322, 105, 0, 106),
  SKIN_OF_UPPER_INNER_QUADRANT_OF_BREAST(FMA, 61426, 107, 0, 210),
  SKIN_OF_UPPER_LIP(SCT, 16251004, 211, 0, 30),
  SKIN_OF_UPPER_OUTER_QUADRANT_OF_LEFT_BREAST(FMA, 61439, 29, 0, 208),
  SKIN_OF_UPPER_PARASPINAL_REGION(DCM, 130302, 209, 0, 232),
  SKIN_OF_UPPER_TRUNK(SCT, 54440003, 232, 0, 0),
  SKIN_OF_VERMILION_PROPER_OF_LOWER_LIP(FMA, 312651, 0, 57, 34),
  SKIN_OF_VERMILION_PROPER_OF_UPPER_LIP(FMA, 312647, 33, 56, 32),
  SKIN_OF_VERTEX_OF_SCALP(SCT, 61719002, 31, 51, 0),
  STERNAL_SKIN(SCT, 244107007, 0, 201, 0),
  SUBMANDIBULAR_TRIANGLE(SCT, 5713008, 39, 0, 40),
  TOOTH(SCT, 38199008, 0, 0, 0),
  VAGINAL_INTROITUS(SCT, 18857001, 0, 506, 0),
  VULVAL_VESTIBULE(SCT, 23213005, 0, 516, 0);

  private final CodingScheme scheme;
  private final String codeValue;
  private final int left;
  private final int middle;
  private final int right;

  SurfacePart(CodingScheme scheme, int codeValue, int left, int middle, int right) {
    this.scheme = scheme;
    this.codeValue = String.valueOf(codeValue);
    this.left = left;
    this.middle = middle;
    this.right = right;
  }

  @Override
  public String getCodeValue() {
    return codeValue;
  }

  @Override
  public String getCodeMeaning() {
    return MesSurface.getString(codeValue);
  }

  @Override
  public CodingScheme getCodingScheme() {
    return scheme;
  }

  @Override
  public String getLegacyCode() {
    return null;
  }

  @Override
  public boolean isPaired() {
    return left != 0;
  }

  public int getLeft() {
    return left;
  }

  public int getMiddle() {
    return middle;
  }

  public int getRight() {
    return right;
  }

  @Override
  public String toString() {
    return getCodeMeaning();
  }

  public static SurfacePart getSurfacePartFromCode(String code) {
    return AnatomicBuilder.getSurfacePartFromCode(code);
  }
}
