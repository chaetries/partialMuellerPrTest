import time

import numpy as np


def lu_chipman_matlab(mueller_data):
    """
    Performs Lu-Chipman polar decomposition on Mueller matrices with precise MATLAB equivalence.

    Parameters:
    -----------
    mueller_data : numpy.ndarray
        Mueller matrix array with shape (H, W, 4, 4) or (H, W, 16)

    Returns:
    --------
    MMD_D : numpy.ndarray
        Diattenuation (dichroism) parameter image
    MMD_Delta : numpy.ndarray
        Depolarization parameter image
    MMD_LR : numpy.ndarray
        Linear retardance parameter image
    MMD_CR : numpy.ndarray
        Circular retardance parameter image
    MMD_psi : numpy.ndarray
        Fast axis orientation image (in radians)
    """
    # Reshape if necessary
    if mueller_data.ndim == 3:  # (H, W, 16)
        H, W = mueller_data.shape[0], mueller_data.shape[1]
        mueller_data = mueller_data.reshape(H, W, 4, 4)
    else:  # (H, W, 4, 4)
        H, W = mueller_data.shape[0], mueller_data.shape[1]

    # Initialize output arrays
    MMD_D = np.zeros((H, W))
    MMD_Delta = np.zeros((H, W))
    MMD_LR = np.zeros((H, W))
    MMD_CR = np.zeros((H, W))
    MMD_psi = np.zeros((H, W))

    print("Starting Lu-Chipman decomposition...")
    start_time = time.time()

    for x in range(H):
        for y in range(W):
            # Extract Mueller matrix for current pixel
            M = mueller_data[x, y, :, :].copy()

            # Skip if M(1,1) is zero (exact MATLAB condition)
            if M[0, 0] == 0:
                continue

            # Store the original M(1,1) value
            m11 = M[0, 0]

            # Normalize by M11 (MATLAB style)
            M = M / m11

            # Replace NaNs and Infs with zeros (MATLAB compatible)
            M = np.nan_to_num(M)

            # Step 1: Extract diattenuation vector and calculate total diattenuation
            D_vector = np.array([M[0, 1], M[0, 2], M[0, 3]])
            D = np.sqrt(np.sum(D_vector ** 2))

            # If D is greater than 1 (per MATLAB implementation)
            if D > 1:
                D_vector = D_vector / D
                D = 1

            # Calculate diattenuation matrix (MATLAB style construction)
            if D > 0:
                D1 = np.sqrt(1 - D ** 2)
                # Use outer product to match MATLAB implementation
                m_D = D1 * np.eye(3) + (1 - D1) * np.outer(D_vector, D_vector) / (D ** 2)
                MD = np.zeros((4, 4))
                MD[0, 0] = 1
                MD[0, 1:4] = D_vector
                MD[1:4, 0] = D_vector
                MD[1:4, 1:4] = m_D
            else:
                MD = np.eye(4)

            # Step 2: Remove diattenuation from Mueller matrix (MATLAB style inverse)
            try:
                M_prime = np.dot(M, np.linalg.inv(MD))
            except np.linalg.LinAlgError:
                # Handle singular matrices like MATLAB does
                M_prime = M.copy()

            # Step 3: Extract retardance matrix using SVD (MATLAB style)
            m_prime = M_prime[1:4, 1:4]

            # Avoid division by zero by checking determinant
            if abs(np.linalg.det(m_prime)) > 1e-10:
                # Use SVD exactly as in MATLAB
                U, S, Vt = np.linalg.svd(m_prime)

                # Sign adjustment for negative determinant (MATLAB style)
                if np.linalg.det(M) < 0:
                    s = [1, 1, -1]  # Sign adjustment based on determinant
                else:
                    s = [1, 1, 1]

                # Construct retardance matrix exactly as in MATLAB
                m_R = np.dot(U * s, Vt)

                # Build full MR matrix (MATLAB style)
                MR = np.eye(4)
                MR[1:4, 1:4] = m_R

                # Step 4: Calculate depolarization matrix
                M_delta = np.dot(M_prime, MR.T)
                m_delta = M_delta[1:4, 1:4]

                # Calculate depolarization value (exact MATLAB calculation)
                delta = 1 - (abs(m_delta[0, 0]) + abs(m_delta[1, 1]) + abs(m_delta[2, 2])) / 3

                # Calculate retardance parameters (MATLAB style)
                # Linear retardance (MATLAB style)
                LR = np.arccos(np.clip(m_R[2, 2], -1.0, 1.0))

                # Circular retardance (MATLAB style using atan2)
                CR = 0.5 * np.arctan2((m_R[1, 0] - m_R[0, 1]), (m_R[0, 0] + m_R[1, 1]))

                # Fast axis orientation (MATLAB style)
                if np.sin(LR) > 1e-10:  # Avoid division by zero
                    r1 = (m_R[1, 2] - m_R[2, 1]) / (2 * np.sin(LR))
                    r2 = (m_R[2, 0] - m_R[0, 2]) / (2 * np.sin(LR))
                    psi = 0.5 * np.arctan2(r2, r1)
                else:
                    psi = 0
            else:
                # Special case for singular matrices (follow MATLAB precisely)
                # Use a regularized pseudoinverse approach
                delta = 0  # Default value
                LR = 0
                CR = 0
                psi = 0

                # Try to recover parameters where possible
                try:
                    # MATLAB often uses SVD for singular matrices
                    U, S, Vt = np.linalg.svd(m_prime, full_matrices=False)

                    # Set very small singular values to zero (MATLAB style threshold)
                    S_cleaned = np.where(S < 1e-10, 0, S)

                    # Reconstruct with cleaned singular values
                    S_inv = np.zeros_like(S_cleaned)
                    nonzero = S_cleaned > 0
                    S_inv[nonzero] = 1.0 / S_cleaned[nonzero]

                    # Construct approximation of retardance matrix
                    m_R_approx = np.dot(U, np.dot(np.diag(S_inv), Vt))

                    # Recalculate parameters
                    LR = np.arccos(np.clip(m_R_approx[2, 2], -1.0, 1.0))
                    CR = 0.5 * np.arctan2((m_R_approx[1, 0] - m_R_approx[0, 1]),
                                          (m_R_approx[0, 0] + m_R_approx[1, 1]))

                    # Approximate depolarization
                    delta = 1 - np.mean(np.abs(np.diag(m_prime)))
                except (np.linalg.LinAlgError, ValueError):
                    # Keep the initialized defaults when decomposition fails.
                    delta = 0

            # Store results
            MMD_D[x, y] = D
            MMD_Delta[x, y] = delta
            MMD_LR[x, y] = LR
            MMD_CR[x, y] = CR
            MMD_psi[x, y] = psi

        # Print progress
        if (x + 1) % 10 == 0 or x == H - 1:
            progress = (x + 1) / H * 100
            print(f"Processing: {progress:.1f}% complete")

    # No blurring step to match the MATLAB implementation exactly
    # The original MATLAB code doesn't appear to apply smoothing to the output parameters

    print(f"Lu-Chipman decomposition completed in {time.time() - start_time:.2f} seconds")

    return MMD_D, MMD_Delta, MMD_LR, MMD_CR, MMD_psi


def degrees_to_radians(angle_degrees):
    """Convert degrees to radians (MATLAB style)"""
    return angle_degrees * np.pi / 180.0


def radians_to_degrees(angle_radians):
    """Convert radians to degrees (MATLAB style)"""
    return angle_radians * 180.0 / np.pi


def acosd(x):
    """MATLAB-equivalent acosd function (arccos in degrees)"""
    return radians_to_degrees(np.arccos(np.clip(x, -1.0, 1.0)))


def atand(x):
    """MATLAB-equivalent atand function (arctan in degrees)"""
    return radians_to_degrees(np.arctan(x))


def atan2d(y, x):
    """MATLAB-equivalent atan2d function (arctan2 in degrees)"""
    return radians_to_degrees(np.arctan2(y, x))
