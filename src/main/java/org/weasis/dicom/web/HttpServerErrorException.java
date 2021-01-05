/*
 * Copyright (c) 2019 Weasis Team and other contributors.
 *
 * This program and the accompanying materials are made available under the terms of the Eclipse
 * Public License 2.0 which is available at http://www.eclipse.org/legal/epl-2.0, or the Apache
 * License, Version 2.0 which is available at https://www.apache.org/licenses/LICENSE-2.0.
 *
 * SPDX-License-Identifier: EPL-2.0 OR Apache-2.0
 */
package org.weasis.dicom.web;

public class HttpServerErrorException extends RuntimeException {

  private static final long serialVersionUID = 1253673551984892314L;

  public HttpServerErrorException(String message) {
    super(message);
  }

  public HttpServerErrorException(String message, Throwable cause) {
    super(message, cause);
  }
}
